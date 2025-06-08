import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Polygon

from color_gamut import rgbv_vertices, rgbv_polygon, rgbcx_vertices, rgbcx_polygon, is_in_gamut, find_closest_boundary_point, xy_to_poly_channels, poly_channels_to_xy
from color_utils import generate_points_in_polygon, loss_function, deltaE_xy
from color_cnn_multi import train_cnn_model_multi, cnn_map_color_points_multi

NUM_IN_POINTS = 200  # 内部点数量             
NUM_BOUNDARY_POINTS = 200  # 边界点数量远大于内部点
EPOCHS = 4000
LEARNING_RATE = 1e-2
FEEDBACK_ROUNDS = 8
FEEDBACK_BATCH = 20

# ========== 敏感性分析参数 ===========
SENS_NUM = 5  # 扰动组数，可根据需要调整
PERTURB_STD = 0.01  # 每个顶点xy扰动标准差

def sample_polygon_boundary(vertices, num_points):
    """均匀采样多边形边界上的点"""
    n = len(vertices)
    edge_lengths = [np.linalg.norm(vertices[i] - vertices[(i+1)%n]) for i in range(n)]
    total_length = sum(edge_lengths)
    points = []
    for _ in range(num_points):
        d = np.random.uniform(0, total_length)
        acc = 0
        for i, l in enumerate(edge_lengths):
            if acc + l >= d:
                t = (d - acc) / l
                pt = vertices[i] + t * (vertices[(i+1)%n] - vertices[i])
                points.append(pt)
                break
            acc += l
    return np.array(points)

def plot_color_mapping(original_points, mapped_points, src_polygon, dst_polygon):
    fig, ax = plt.subplots(figsize=(8, 6))
    # 可视化原色域和目标色域的面积重叠
    src_poly = Polygon(src_polygon.vertices, facecolor='blue', alpha=0.15, edgecolor='blue', lw=2, label='RGBV Area')
    dst_poly = Polygon(dst_polygon.vertices, facecolor='red', alpha=0.15, edgecolor='red', lw=2, label='RGBCX Area')
    ax.add_patch(src_poly)
    ax.add_patch(dst_poly)
    src_patch = PathPatch(src_polygon, facecolor='none', edgecolor='blue', lw=2, label='RGBV Gamut')
    ax.add_patch(src_patch)
    dst_patch = PathPatch(dst_polygon, facecolor='none', edgecolor='red', lw=2, label='RGBCX Gamut')
    ax.add_patch(dst_patch)
    ax.scatter(original_points[:, 0], original_points[:, 1], c='blue', label='Original Points', s=10, alpha=0.8)
    ax.scatter(mapped_points[:, 0], mapped_points[:, 1], c='red', label='Mapped Points', s=10, alpha=0.8)
    for i in range(len(original_points)):
        ax.plot([original_points[i, 0], mapped_points[i, 0]],
                [original_points[i, 1], mapped_points[i, 1]],
                'k--', alpha=0.2, linewidth=0.5)
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 0.9)
    ax.set_xlabel('x Chromaticity')
    ax.set_ylabel('y Chromaticity')
    ax.set_title('Color Mapping from RGBV to RGBCX')
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
        perturbed_vertices = perturb_vertices(rgbcx_vertices, std=PERTURB_STD)
        perturbed_polygon = type(rgbcx_polygon)(perturbed_vertices)
        # 2. 生成目标点（xy）
        train_targets = []
        for pt in all_points:
            if is_in_gamut(pt, perturbed_polygon):
                train_targets.append(pt)
            else:
                train_targets.append(find_closest_boundary_point(pt, perturbed_polygon))
        train_targets = np.array(train_targets)
        # 3. xy转五通道分量
        train_targets_channels = np.array([xy_to_poly_channels(xy, perturbed_vertices) for xy in train_targets])
        # 4. 训练CNN
        cnn_model = train_cnn_model_multi(
            all_points_channels, train_targets_channels,
            vertices=perturbed_vertices,
            epochs=EPOCHS, lr=LEARNING_RATE,
            feedback_rounds=FEEDBACK_ROUNDS, feedback_batch=FEEDBACK_BATCH,
            sample_weights=sample_weights
        )
        # 5. 用CNN映射
        mapped_channels = cnn_map_color_points_multi(all_points_channels, cnn_model)
        mapped_points = np.array([poly_channels_to_xy(c, perturbed_vertices) for c in mapped_channels])
        # 6. 计算deltaE
        deltaEs = np.array([deltaE_xy(all_points[j], mapped_points[j]) for j in range(len(all_points))])
        all_deltaEs.append(deltaEs)
        print(f"第{i+1}组扰动完成，平均deltaE: {np.mean(deltaEs):.4f}")
    all_deltaEs = np.array(all_deltaEs)  # shape: (SENS_NUM, N)
    std_per_point = np.std(all_deltaEs, axis=0)  # 每个点的std
    mean_std = np.mean(std_per_point)
    print(f"\n敏感性分析：{SENS_NUM}组扰动下，所有点的deltaE标准差均值为: {mean_std:.6f}")
    return all_deltaEs, std_per_point, mean_std

if __name__ == "__main__":
    # 采样RGBV色域内点和大量边界点
    in_points = generate_points_in_polygon(rgbv_vertices, NUM_IN_POINTS)
    boundary_points = sample_polygon_boundary(rgbv_vertices, NUM_BOUNDARY_POINTS)
    all_points = np.vstack([in_points, boundary_points])

    # xy转为四通道分量
    in_points_channels = np.array([xy_to_poly_channels(xy, rgbv_vertices) for xy in in_points])
    boundary_points_channels = np.array([xy_to_poly_channels(xy, rgbv_vertices) for xy in boundary_points])
    all_points_channels = np.vstack([in_points_channels, boundary_points_channels])

    # 训练数据
    train_points_channels = np.vstack([in_points_channels, boundary_points_channels])
    train_targets = []
    sample_weights = []
    for pt in in_points:
        if is_in_gamut(pt, rgbcx_polygon):
            train_targets.append(pt)
        else:
            train_targets.append(find_closest_boundary_point(pt, rgbcx_polygon))
        sample_weights.append(1.0)
    for pt in boundary_points:
        if is_in_gamut(pt, rgbcx_polygon):
            train_targets.append(pt)
        else:
            train_targets.append(find_closest_boundary_point(pt, rgbcx_polygon))
        sample_weights.append(10.0)
    train_targets = np.array(train_targets)
    sample_weights = np.array(sample_weights)

    # 目标xy转为五通道分量
    train_targets_channels = np.array([xy_to_poly_channels(xy, rgbcx_vertices) for xy in train_targets])

    # 训练CNN模型（四通道到五通道）
    cnn_model = train_cnn_model_multi(
        train_points_channels, train_targets_channels,
        vertices=rgbcx_vertices,
        epochs=EPOCHS, lr=LEARNING_RATE,
        feedback_rounds=FEEDBACK_ROUNDS, feedback_batch=FEEDBACK_BATCH,
        sample_weights=sample_weights
    )

    # 用CNN映射（输入四通道，输出五通道，再映射到xy）
    mapped_channels = cnn_map_color_points_multi(all_points_channels, cnn_model)
    mapped_points = np.array([poly_channels_to_xy(c, rgbcx_vertices) for c in mapped_channels])
    losses = np.linalg.norm(all_points - mapped_points, axis=1)

    # 可视化
    plot_color_mapping(all_points, mapped_points, rgbv_polygon, rgbcx_polygon)

    print(f"总测试点数: {len(all_points)}")
    print(f"平均映射损失: {np.mean(losses):.4f}")
    print("\n前5个点映射示例:")
    for i in range(5):
        rgbv = all_points_channels[i]
        rgbcx = mapped_channels[i]
        print(f"原始点xy: {all_points[i]}, 映射点xy: {mapped_points[i]}, 损失: {losses[i]:.4f}")
        print(f"原始RGBV: {rgbv}, 映射RGBCX: {rgbcx}")
    # ====== 敏感性分析 ======
    all_deltaEs, std_per_point, mean_std = sensitivity_analysis()

"""
# 敏感性分析报告（RGBV→RGBCX）

本实验针对RGBCX色域顶点（rgbcx_vertices）进行了敏感性分析，具体内容如下：

## 分析方法
- 对RGBCX五边形的五个顶点分别施加高斯扰动（标准差为0.01），共生成5组不同的目标色域顶点。
- 每组扰动下，均重新训练色域映射CNN模型，并对RGBV色域内及边界采样点进行映射。
- 计算每组扰动下所有采样点的deltaE（CIEDE2000色差），并统计每个点在5组扰动下的deltaE标准差。
- 最终以所有点的deltaE标准差均值作为敏感性分析指标。

## 实验参数
- 扰动组数 SENS_NUM = 5
- 顶点扰动标准差 PERTURB_STD = 0.01

## 实验结果
- 第5组扰动完成，平均deltaE: 3.8432
- 5组扰动下，所有点的deltaE标准差均值为: 1.187514

## 结论
- 在当前扰动范围和模型设定下，RGBCX色域顶点的小幅变化会对最终色彩映射的deltaE产生一定影响，但整体标准差仍处于可控范围，模型具有一定的鲁棒性。
"""
