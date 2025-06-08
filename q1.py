import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
import copy
from matplotlib.path import Path

from color_gamut import bt2020_vertices, bt2020_triangle, poly_channels_to_xy, std_rgb_vertices, std_rgb_triangle, is_in_gamut, find_closest_boundary_point, xy_to_poly_channels
from color_utils import generate_points_in_triangle, generate_points_outside_triangle, deltaE_xy, loss_function
from color_cnn import train_cnn_model, cnn_map_color_points

# ==================== 可调参数 ====================
NUM_IN_POINTS = 100         # BT2020色域内采样点数
EPOCHS = 3000                 # CNN训练轮数
LEARNING_RATE = 1e-2         # 学习率
FEEDBACK_ROUNDS = 0          # 反馈训练轮数
FEEDBACK_BATCH = 10          # 每轮反馈采样点数

# ========== 敏感性分析参数 ===========
SENS_NUM = 5  # 扰动组数，可根据需要调整
PERTURB_STD = 0.01  # 每个顶点xy扰动标准差


def batch_map_color_points(points, gamut_triangle, use_cnn=False, cnn_model=None):
    """
    批量处理颜色点，返回映射后的点和损失值
    若use_cnn为True，则用CNN模型映射，否则用传统方法
    """
    if use_cnn and cnn_model is not None:
        mapped_points = cnn_map_color_points(points, cnn_model)
        losses = np.linalg.norm(points - mapped_points, axis=1)
        return mapped_points, losses
    mapped_points = []
    losses = []
    for point in points:
        if is_in_gamut(point, gamut_triangle):
            mapped_points.append(point)
            losses.append(0.0)
        else:
            closest_point = find_closest_boundary_point(point, gamut_triangle)
            mapped_points.append(closest_point)
            losses.append(loss_function(point, closest_point))
    return np.array(mapped_points), np.array(losses)

def plot_color_mapping(original_points, mapped_points, src_triangle, dst_triangle):
    """绘制色度图与映射路径"""
    fig, ax = plt.subplots(figsize=(8, 6))
    src_patch = PathPatch(src_triangle, facecolor='none', edgecolor='blue', lw=2, label='BT2020 Gamut')
    ax.add_patch(src_patch)
    dst_patch = PathPatch(dst_triangle, facecolor='none', edgecolor='red', lw=2, label='Standard RGB Gamut')
    ax.add_patch(dst_patch)
    ax.scatter(original_points[:, 0], original_points[:, 1], c='blue', label='Original Points', s=20, alpha=0.8)
    ax.scatter(mapped_points[:, 0], mapped_points[:, 1], c='red', label='Mapped Points', s=20, alpha=0.8)
    for i in range(len(original_points)):
        ax.plot([original_points[i, 0], mapped_points[i, 0]],
                [original_points[i, 1], mapped_points[i, 1]],
                'k--', alpha=0.5, linewidth=1)
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 0.9)
    ax.set_xlabel('x Chromaticity')
    ax.set_ylabel('y Chromaticity')
    ax.set_title('Color Mapping from BT2020 to Standard RGB')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def perturb_vertices(vertices, std=0.01):
    """对色域顶点加高斯扰动，返回新顶点"""
    perturbed = vertices + np.random.normal(0, std, vertices.shape)
    return perturbed

def sensitivity_analysis():
    all_deltaEs = []
    for i in range(SENS_NUM):
        # 1. 扰动目标色域顶点
        perturbed_vertices = perturb_vertices(std_rgb_vertices, std=PERTURB_STD)
        perturbed_triangle = Path(perturbed_vertices)
        # 2. 生成目标点（xy）
        train_targets = []
        for pt in train_points:
            if is_in_gamut(pt, perturbed_triangle):
                train_targets.append(pt)
            else:
                train_targets.append(find_closest_boundary_point(pt, perturbed_triangle))
        train_targets = np.array(train_targets)
        # 3. xy转三通道分量
        train_targets_channels = np.array([xy_to_poly_channels(xy, perturbed_vertices) for xy in train_targets])
        # 4. 训练CNN
        cnn_model = train_cnn_model(
            train_points_channels, train_targets_channels,
            epochs=EPOCHS, lr=LEARNING_RATE,
            feedback_rounds=FEEDBACK_ROUNDS, feedback_batch=FEEDBACK_BATCH
        )
        # 5. 用CNN映射
        mapped_channels, _ = batch_map_color_points(train_points_channels, perturbed_triangle, use_cnn=True, cnn_model=cnn_model)
        mapped_points = np.array([poly_channels_to_xy(c, perturbed_vertices) for c in mapped_channels])
        # 6. 计算deltaE
        deltaEs = np.array([deltaE_xy(train_points[j], mapped_points[j]) for j in range(len(train_points))])
        all_deltaEs.append(deltaEs)
        print(f"第{i+1}组扰动完成，平均deltaE: {np.mean(deltaEs):.4f}")
    all_deltaEs = np.array(all_deltaEs)  # shape: (SENS_NUM, N)
    std_per_point = np.std(all_deltaEs, axis=0)  # 每个点的std
    mean_std = np.mean(std_per_point)
    print(f"\n敏感性分析：{SENS_NUM}组扰动下，所有点的deltaE标准差均值为: {mean_std:.6f}")
    return all_deltaEs, std_per_point, mean_std

if __name__ == "__main__":
    # 生成测试数据
    in_points = generate_points_in_triangle(bt2020_vertices, NUM_IN_POINTS)
    all_points = np.vstack([in_points])

    # 生成训练数据（xy转为三通道分量）
    train_points = np.vstack([in_points])
    train_targets = []
    for pt in train_points:
        if is_in_gamut(pt, std_rgb_triangle):
            train_targets.append(pt)
        else:
            train_targets.append(find_closest_boundary_point(pt, std_rgb_triangle))
    train_targets = np.array(train_targets)

    # 新增：xy转为三通道分量
    train_points_channels = np.array([xy_to_poly_channels(xy, bt2020_vertices) for xy in train_points])
    train_targets_channels = np.array([xy_to_poly_channels(xy, std_rgb_vertices) for xy in train_targets])

    # 训练CNN模型（输入输出均为三通道分量）
    cnn_model = train_cnn_model(
        train_points_channels, train_targets_channels,
        epochs=EPOCHS, lr=LEARNING_RATE,
        feedback_rounds=FEEDBACK_ROUNDS, feedback_batch=FEEDBACK_BATCH
    )

    # 用CNN映射（输入三通道，输出三通道，再映射到xy）
    mapped_channels, _ = batch_map_color_points(train_points_channels, std_rgb_triangle, use_cnn=True, cnn_model=cnn_model)
    mapped_points = np.array([poly_channels_to_xy(c, std_rgb_vertices) for c in mapped_channels])
    losses = np.linalg.norm(train_points - mapped_points, axis=1)

    # 随机选取100个点用于可视化
    idx = np.random.choice(len(train_points), size=100, replace=False)
    plot_color_mapping(train_points[idx], mapped_points[idx], bt2020_triangle, std_rgb_triangle)

    # 打印部分结果（转换为RGB向量后输出）
    print(f"总测试点数: {len(train_points)}")
    print(f"平均映射损失: {np.mean(losses):.4f}")
    print("\n前5个点映射示例:")
    for i in range(5):
        # gamut中的xy_to_poly_channels分别反解为RGB向量
        rgb_in = xy_to_poly_channels(train_points[i], bt2020_vertices)
        rgb_out = xy_to_poly_channels(mapped_points[i], std_rgb_vertices)
        print(f"原始点xy: {train_points[i]}, 映射点xy: {mapped_points[i]}, 损失: {losses[i]:.4f}")
        print(f"原始RGB: {rgb_in}, 映射RGB: {rgb_out}")

    # ====== 敏感性分析 ======
    all_deltaEs, std_per_point, mean_std = sensitivity_analysis()