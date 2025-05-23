import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch

from color_gamut import bt2020_vertices, bt2020_triangle, std_rgb_vertices, std_rgb_triangle, is_in_gamut, find_closest_boundary_point
from color_utils import generate_points_in_triangle, generate_points_outside_triangle, loss_function
from color_cnn import train_cnn_model, cnn_map_color_points

# ==================== 可调参数 ====================
NUM_IN_POINTS = 1000         # BT2020色域内采样点数
EPOCHS = 300                 # CNN训练轮数
LEARNING_RATE = 1e-2         # 学习率
FEEDBACK_ROUNDS = 5          # 反馈训练轮数
FEEDBACK_BATCH = 10          # 每轮反馈采样点数

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

if __name__ == "__main__":
    # 生成测试数据
    in_points = generate_points_in_triangle(bt2020_vertices, NUM_IN_POINTS)
    all_points = np.vstack([in_points])

    # 生成训练数据
    train_points = np.vstack([in_points])
    train_targets = []
    for pt in train_points:
        if is_in_gamut(pt, std_rgb_triangle):
            train_targets.append(pt)
        else:
            train_targets.append(find_closest_boundary_point(pt, std_rgb_triangle))
    train_targets = np.array(train_targets)

    # 训练CNN模型
    cnn_model = train_cnn_model(
        train_points, train_targets,
        epochs=EPOCHS, lr=LEARNING_RATE,
        feedback_rounds=FEEDBACK_ROUNDS, feedback_batch=FEEDBACK_BATCH
    )

    # 用CNN映射
    mapped_points, losses = batch_map_color_points(all_points, std_rgb_triangle, use_cnn=True, cnn_model=cnn_model)

    # 展示CNN映射函数的权重和偏置
    print("\nCNN模型结构与参数：")
    print(cnn_model)
    for name, param in cnn_model.named_parameters():
        print(f"{name}: {param.data.numpy()}")

    # 可视化结果
    plot_color_mapping(all_points, mapped_points, bt2020_triangle, std_rgb_triangle)

    # 打印部分结果
    print(f"总测试点数: {len(all_points)}")
    print(f"平均映射损失: {np.mean(losses):.4f}")
    print("\n前5个点映射示例:")
    for i in range(5):
        print(f"原始点: {all_points[i]}, 映射点: {mapped_points[i]}, 损失: {losses[i]:.4f}")