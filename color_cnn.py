import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from color_utils import deltaE_xy
from color_gamut import is_in_gamut, find_closest_boundary_point, std_rgb_triangle

class SimpleColorCNN(nn.Module):
    """更复杂的4层全连接神经网络用于xy色度点映射"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.net(x)

def train_cnn_model(train_x, train_y, epochs=200, lr=1e-2, feedback_rounds=5, feedback_batch=10, sample_weights=None):
    """
    训练CNN模型，主损失为MSE，监控deltaE，输出不在目标色域内时加惩罚。
    支持sample_weights对样本加权。
    """
    device = torch.device('cpu')
    model = SimpleColorCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    x_tensor = torch.tensor(train_x, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(train_y, dtype=torch.float32).to(device)
    if sample_weights is not None:
        weights_tensor = torch.tensor(sample_weights, dtype=torch.float32).to(device)
    else:
        weights_tensor = torch.ones(len(train_x), dtype=torch.float32).to(device)
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(x_tensor)
        mse = ((out - y_tensor) ** 2).sum(dim=1)
        mse_loss = (mse * weights_tensor).mean()
        out_np = out.detach().cpu().numpy()
        y_np = y_tensor.detach().cpu().numpy()
        penalty = 0.0
        for i in range(out_np.shape[0]):
            if not is_in_gamut(out_np[i], std_rgb_triangle):
                boundary_pt = find_closest_boundary_point(out_np[i], std_rgb_triangle)
                penalty += np.linalg.norm(out_np[i] - boundary_pt) * float(weights_tensor[i])
        loss = mse_loss + 50.0 * penalty / weights_tensor.sum()  # 惩罚系数进一步提高
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}, weighted MSE: {mse_loss.item():.6f}")
    # 反馈训练
    for round in range(feedback_rounds):
        idx = np.random.choice(len(train_x), feedback_batch, replace=False)
        test_x = train_x[idx]
        with torch.no_grad():
            pred = model(torch.tensor(test_x, dtype=torch.float32).to(device)).cpu().numpy()
        feedback_x = []
        feedback_y = []
        for i in range(feedback_batch):
            if not is_in_gamut(pred[i], std_rgb_triangle):
                feedback_x.append(test_x[i])
                feedback_y.append(find_closest_boundary_point(pred[i], std_rgb_triangle))
        if feedback_x:
            fx = torch.tensor(np.array(feedback_x), dtype=torch.float32).to(device)
            fy = torch.tensor(np.array(feedback_y), dtype=torch.float32).to(device)
            for _ in range(10):
                optimizer.zero_grad()
                out = model(fx)
                mse_loss = nn.functional.mse_loss(out, fy)
                out_np = out.detach().cpu().numpy()
                fy_np = fy.detach().cpu().numpy()
                deltaE_loss = 0.0
                for i in range(out_np.shape[0]):
                    deltaE_loss += deltaE_xy(out_np[i], fy_np[i])
                deltaE_loss = deltaE_loss / out_np.shape[0]
                penalty = 0.0
                for i in range(out_np.shape[0]):
                    if not is_in_gamut(out_np[i], std_rgb_triangle):
                        boundary_pt = find_closest_boundary_point(out_np[i], std_rgb_triangle)
                        penalty += np.linalg.norm(out_np[i] - boundary_pt)
                loss = mse_loss + 20.0 * penalty / out_np.shape[0]
                loss.backward()
                optimizer.step()
            print(f"Feedback round {round+1}/{feedback_rounds}, feedback deltaE: {deltaE_loss:.6f}")
    return model

def cnn_map_color_points(points, model):
    """用训练好的CNN模型批量映射xy点"""
    device = torch.device('cpu')
    x_tensor = torch.tensor(points, dtype=torch.float32).to(device)
    with torch.no_grad():
        mapped = model(x_tensor).cpu().numpy()
    return mapped
