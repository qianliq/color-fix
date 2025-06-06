import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from color_utils import deltaE_xy
from color_gamut import poly_channels_to_xy, std_rgb_vertices, is_in_gamut, find_closest_boundary_point
from matplotlib.path import Path

class SimpleColorCNN(nn.Module):
    """三通道到三通道的全连接网络"""
    def __init__(self, channels=3):
        super().__init__()
        self.channels = channels
        self.net = nn.Sequential(
            nn.Linear(channels, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, channels)
        )
    def forward(self, x):
        out = self.net(x)
        # softmax保证输出为概率分布且和为1
        return torch.softmax(out, dim=1)

def torch_poly_channels_to_xy(channels, vertices):
    """
    torch实现的多通道到xy的线性映射
    channels: (N, C) tensor
    vertices: (C, 2) numpy array or tensor
    返回: (N, 2) tensor
    """
    if not isinstance(vertices, torch.Tensor):
        vertices = torch.tensor(vertices, dtype=channels.dtype, device=channels.device)
    return torch.matmul(channels, vertices)

def torch_xy_to_XYZ(xy, Y=1.0):
    """
    torch实现的xy转XYZ
    xy: (N,2) tensor
    返回: (N,3) tensor
    """
    x = xy[:, 0]
    y = xy[:, 1]
    X = (x / y) * Y
    Z = ((1 - x - y) / y) * Y
    XYZ = torch.stack([X, torch.full_like(X, Y), Z], dim=1)
    XYZ[y == 0] = 0
    return XYZ

def torch_XYZ_to_Lab(XYZ):
    """
    torch实现的XYZ转Lab（近似，D65白点，简化版）
    仅用于loss反向传播，精度略低于skimage/scipy
    """
    # D65白点
    Xn = 0.95047
    Yn = 1.0
    Zn = 1.08883
    X = XYZ[:, 0] / Xn
    Y = XYZ[:, 1] / Yn
    Z = XYZ[:, 2] / Zn

    def f(t):
        delta = 6/29
        return torch.where(t > delta**3, torch.pow(t, 1/3), t/(3*delta**2) + 4/29)

    fx = f(X)
    fy = f(Y)
    fz = f(Z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return torch.stack([L, a, b], dim=1)

def torch_rad2deg(x):
    """torch实现的弧度转角度"""
    return x * (180.0 / np.pi)

def torch_deg2rad(x):
    """torch实现的角度转弧度"""
    return x * (np.pi / 180.0)

def torch_deltaE2000(Lab1, Lab2):
    """
    torch实现的CIEDE2000色差（近似，适合反向传播）
    Lab1, Lab2: (N,3) tensor
    返回: (N,) tensor
    """
    L1, a1, b1 = Lab1[:, 0], Lab1[:, 1], Lab1[:, 2]
    L2, a2, b2 = Lab2[:, 0], Lab2[:, 1], Lab2[:, 2]

    avg_L = (L1 + L2) / 2.0
    C1 = torch.sqrt(a1 ** 2 + b1 ** 2)
    C2 = torch.sqrt(a2 ** 2 + b2 ** 2)
    avg_C = (C1 + C2) / 2.0

    G = 0.5 * (1 - torch.sqrt((avg_C ** 7) / (avg_C ** 7 + 25 ** 7 + 1e-8)))
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = torch.sqrt(a1p ** 2 + b1 ** 2)
    C2p = torch.sqrt(a2p ** 2 + b2 ** 2)
    avg_Cp = (C1p + C2p) / 2.0

    h1p = torch.atan2(b1, a1p)
    h1p = h1p % (2 * np.pi)
    h2p = torch.atan2(b2, a2p)
    h2p = h2p % (2 * np.pi)

    delt_Lp = L2 - L1
    delt_Cp = C2p - C1p

    delt_hp = h2p - h1p
    delt_hp = delt_hp - (2 * np.pi) * torch.round(delt_hp / (2 * np.pi))
    delt_Hp = 2 * torch.sqrt(C1p * C2p) * torch.sin(delt_hp / 2.0)

    avg_Lp = (L1 + L2) / 2.0
    avg_Hp = (h1p + h2p) / 2.0
    avg_Hp = avg_Hp - (2 * np.pi) * (torch.abs(h1p - h2p) > np.pi).float() * 0.5

    T = 1 - 0.17 * torch.cos(avg_Hp - np.deg2rad(30)) + \
        0.24 * torch.cos(2 * avg_Hp) + \
        0.32 * torch.cos(3 * avg_Hp + np.deg2rad(6)) - \
        0.20 * torch.cos(4 * avg_Hp - np.deg2rad(63))

    delt_ro = 30 * torch.exp(-((torch_rad2deg(avg_Hp) - 275) / 25) ** 2)
    Rc = 2 * torch.sqrt((avg_Cp ** 7) / (avg_Cp ** 7 + 25 ** 7 + 1e-8))
    Sl = 1 + ((0.015 * (avg_Lp - 50) ** 2) / torch.sqrt(20 + (avg_Lp - 50) ** 2))
    Sc = 1 + 0.045 * avg_Cp
    Sh = 1 + 0.015 * avg_Cp * T
    Rt = -torch.sin(2 * torch_deg2rad(delt_ro)) * Rc

    dE = torch.sqrt(
        (delt_Lp / Sl) ** 2 +
        (delt_Cp / Sc) ** 2 +
        (delt_Hp / Sh) ** 2 +
        Rt * (delt_Cp / Sc) * (delt_Hp / Sh)
    )
    return dE

def train_cnn_model(train_x, train_y, vertices=std_rgb_vertices, epochs=200, lr=1e-2, feedback_rounds=5, feedback_batch=10, sample_weights=None):
    """
    训练CNN模型，输入输出均为三通道，损失为deltaE2000近似。
    train_x, train_y: shape (N, 3)，每行为通道分量，和为1
    vertices: shape (3,2)，色域顶点
    """
    device = torch.device('cpu')
    model = SimpleColorCNN(channels=train_x.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    x_tensor = torch.tensor(train_x, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(train_y, dtype=torch.float32).to(device)
    if sample_weights is not None:
        weights_tensor = torch.tensor(sample_weights, dtype=torch.float32).to(device)
    else:
        weights_tensor = torch.ones(len(train_x), dtype=torch.float32).to(device)
    vertices_tensor = torch.tensor(vertices, dtype=torch.float32).to(device)
    eps = 1e-7  # 防止除零和log负数

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(x_tensor)
        # 防止softmax输出为0，导致后续nan
        out = torch.clamp(out, min=eps, max=1.0)
        out = out / (out.sum(dim=1, keepdim=True) + eps)
        # torch实现xy映射
        out_xy = torch_poly_channels_to_xy(out, vertices_tensor)
        y_xy = torch_poly_channels_to_xy(y_tensor, vertices_tensor)
        # torch实现Lab
        out_XYZ = torch_xy_to_XYZ(out_xy)
        y_XYZ = torch_xy_to_XYZ(y_xy)
        out_Lab = torch_XYZ_to_Lab(out_XYZ)
        y_Lab = torch_XYZ_to_Lab(y_XYZ)
        # torch实现deltaE2000近似
        deltaE = torch_deltaE2000(out_Lab, y_Lab)
        # 防止nan
        deltaE = torch.where(torch.isnan(deltaE), torch.zeros_like(deltaE), deltaE)
        loss = torch.mean(deltaE * weights_tensor)
        # 输出不在色域内时加惩罚（此处仍用numpy判断）
        out_xy_np = out_xy.detach().cpu().numpy()
        penalty = 0.0
        for i in range(out_xy_np.shape[0]):
            if not is_in_gamut(out_xy_np[i], Path(vertices)):
                boundary_pt = find_closest_boundary_point(out_xy_np[i], Path(vertices))
                if boundary_pt is not None:
                    penalty += np.linalg.norm(out_xy_np[i] - boundary_pt) * float(weights_tensor[i])
        loss = loss + 50.0 * penalty / (weights_tensor.sum() + eps)
        # 防止loss为nan
        if torch.isnan(loss):
            print(f"Epoch {epoch+1}: loss is nan, break.")
            break
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}, weighted deltaE2000: {loss.item():.6f}")
    # 反馈训练
    for round in range(feedback_rounds):
        idx = np.random.choice(len(train_x), feedback_batch, replace=False)
        test_x = train_x[idx]
        with torch.no_grad():
            pred = model(torch.tensor(test_x, dtype=torch.float32).to(device)).cpu().numpy()
        pred_xy = np.array([poly_channels_to_xy(p, vertices) for p in pred])
        feedback_x = []
        feedback_y = []
        for i in range(feedback_batch):
            if not is_in_gamut(pred_xy[i], Path(vertices)):
                feedback_x.append(test_x[i])
                # 反馈目标为边界点的通道分量
                boundary_xy = find_closest_boundary_point(pred_xy[i], Path(vertices))
                from color_gamut import xy_to_poly_channels
                if boundary_xy is not None:
                    boundary_channels = xy_to_poly_channels(boundary_xy, vertices)
                    feedback_y.append(boundary_channels)
        if feedback_x and feedback_y:
            fx = torch.tensor(np.array(feedback_x), dtype=torch.float32).to(device)
            fy = torch.tensor(np.array(feedback_y), dtype=torch.float32).to(device)
            for _ in range(10):
                optimizer.zero_grad()
                out = model(fx)
                out_xy = torch_poly_channels_to_xy(out, vertices_tensor)
                fy_xy = torch_poly_channels_to_xy(fy, vertices_tensor)
                out_XYZ = torch_xy_to_XYZ(out_xy)
                fy_XYZ = torch_xy_to_XYZ(fy_xy)
                out_Lab = torch_XYZ_to_Lab(out_XYZ)
                fy_Lab = torch_XYZ_to_Lab(fy_XYZ)
                deltaE = torch_deltaE2000(out_Lab, fy_Lab)
                loss = torch.mean(deltaE)
                out_xy_np = out_xy.detach().cpu().numpy()
                penalty = 0.0
                for i in range(out_xy_np.shape[0]):
                    if not is_in_gamut(out_xy_np[i], Path(vertices)):
                        boundary_pt = find_closest_boundary_point(out_xy_np[i], Path(vertices))
                        if boundary_pt is not None:
                            penalty += np.linalg.norm(out_xy_np[i] - boundary_pt)
                loss = loss + 20.0 * penalty / out_xy.shape[0]
                loss.backward()
                optimizer.step()
            print(f"Feedback round {round+1}/{feedback_rounds}, feedback deltaE2000 (approx): {loss.item():.6f}")
    return model

def cnn_map_color_points(points, model):
    """用训练好的CNN模型批量映射三通道点"""
    device = torch.device('cpu')
    x_tensor = torch.tensor(points, dtype=torch.float32).to(device)
    with torch.no_grad():
        mapped = model(x_tensor).cpu().numpy()
    return mapped
