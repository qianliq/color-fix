import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from color_gamut import poly_channels_to_xy, std_rgb_vertices
from matplotlib.path import Path

torch.set_num_threads(8)
torch.set_num_interop_threads(4)

class ResidualColorCNN(nn.Module):
    """
    三通道到三通道的残差全连接网络
    - 输入: 归一化RGB三通道 (和为1，非负)
    - 输出: 归一化RGB三通道 (和为1，非负，softmax保证)
    - 残差结构: output = input + f(input)
    - softmax: 保证输出和为1且非负，适合色度空间映射
    """
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
        out = x + out  # 残差结构
        out = torch.softmax(out, dim=1)  # 输出归一化，和为1且非负
        return out

def rgb_to_xyz_torch(rgb):
    # 输入: 归一化RGB (N,3)
    # 输出: XYZ (N,3)
    mask = (rgb > 0.04045)
    rgb_lin = torch.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    M = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], dtype=rgb.dtype, device=rgb.device)
    xyz = torch.matmul(rgb_lin, M.T)
    return xyz

def xyz_to_lab_torch(xyz):
    # 输入: XYZ (N,3)
    # 输出: Lab (N,3)
    Xn = 0.95047
    Yn = 1.0
    Zn = 1.08883
    x = xyz[:, 0] / Xn
    y = xyz[:, 1] / Yn
    z = xyz[:, 2] / Zn
    def f(t):
        delta = 6/29
        return torch.where(t > delta**3, torch.pow(t, 1/3), t/(3*delta**2) + 4/29)
    fx = f(x)
    fy = f(y)
    fz = f(z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return torch.stack([L, a, b], dim=1)

def deltaE2000_torch(Lab1, Lab2):
    # 输入: Lab1, Lab2 (N,3)
    # 输出: deltaE2000色差 (N,)
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
    delt_ro = 30 * torch.exp(-((avg_Hp * 180.0 / np.pi - 275) / 25) ** 2)
    Rc = 2 * torch.sqrt((avg_Cp ** 7) / (avg_Cp ** 7 + 25 ** 7 + 1e-8))
    Sl = 1 + ((0.015 * (avg_Lp - 50) ** 2) / torch.sqrt(20 + (avg_Lp - 50) ** 2))
    Sc = 1 + 0.045 * avg_Cp
    Sh = 1 + 0.015 * avg_Cp * T
    Rt = -torch.sin(2 * np.pi * delt_ro / 180.0) * Rc
    dE = torch.sqrt(
        (delt_Lp / Sl) ** 2 +
        (delt_Cp / Sc) ** 2 +
        (delt_Hp / Sh) ** 2 +
        Rt * (delt_Cp / Sc) * (delt_Hp / Sh)
    )
    return dE

def train_cnn_model(
    train_x, train_y, epochs=200, lr=1e-3,
    feedback_rounds=0, feedback_batch=10, sample_weights=None
):
    """
    训练三通道到三通道的CNN
    输入: train_x (N,3) 归一化RGB
    目标: train_y (N,3) 归一化RGB
    损失: deltaE2000(Lab(输出), Lab(目标)) 的加权平均
    归一化: 输出经过softmax和clamp，保证和为1且非负
    """
    device = torch.device('cpu')
    model = ResidualColorCNN(channels=train_x.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    x_tensor = torch.tensor(train_x, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(train_y, dtype=torch.float32).to(device)
    if sample_weights is not None:
        weights_tensor = torch.tensor(sample_weights, dtype=torch.float32).to(device)
    else:
        weights_tensor = torch.ones(len(train_x), dtype=torch.float32).to(device)
    eps = 1e-7

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(x_tensor)  # [输入] -> [网络] -> [softmax归一化输出]
        out = torch.clamp(out, min=eps, max=1.0)
        out = out / (out.sum(dim=1, keepdim=True) + eps)  # 再归一化，防止数值漂移
        # 用gamut的poly_channels_to_xy函数转xy（可用于可视化/惩罚）
        out_np = out.detach().cpu().numpy()
        y_np = y_tensor.detach().cpu().numpy()
        out_xy = np.array([poly_channels_to_xy(c, std_rgb_vertices) for c in out_np])
        y_xy = np.array([poly_channels_to_xy(c, std_rgb_vertices) for c in y_np])
        out_xy_tensor = torch.tensor(out_xy, dtype=torch.float32).to(device)
        y_xy_tensor = torch.tensor(y_xy, dtype=torch.float32).to(device)
        # [目标最小化] 损失为Lab空间deltaE2000色差
        out_xyz = rgb_to_xyz_torch(out)
        y_xyz = rgb_to_xyz_torch(y_tensor)
        out_lab = xyz_to_lab_torch(out_xyz)
        y_lab = xyz_to_lab_torch(y_xyz)
        deltaE = deltaE2000_torch(out_lab, y_lab)
        if torch.isnan(deltaE).any() or torch.isinf(deltaE).any():
            print(f"[DEBUG][Epoch {epoch+1}] deltaE contains nan/inf")
            print("deltaE min:", deltaE.min().item(), "max:", deltaE.max().item())
            print("deltaE sample:", deltaE[:10])
            break
        loss = torch.mean(deltaE * weights_tensor)  # [目标] 最小化加权deltaE2000
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[DEBUG][Epoch {epoch+1}] loss is nan/inf")
            print("loss value:", loss.item())
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}, weighted deltaE2000: {loss.item():.6f}")
    return model

def cnn_map_color_points(points, model):
    """
    推理接口
    输入: points (N,3) 归一化RGB
    输出: (N,3) 归一化RGB, softmax归一化后输出
    """
    device = torch.device('cpu')
    x_tensor = torch.tensor(points, dtype=torch.float32).to(device)
    with torch.no_grad():
        mapped = model(x_tensor).cpu().numpy()
        mapped = np.nan_to_num(mapped, nan=0.0, posinf=1.0, neginf=0.0)
        mapped = np.clip(mapped, 0, 1)
    return mapped
