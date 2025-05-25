import numpy as np
from color_gamut import std_rgb_vertices, poly_channels_to_xy, xy_to_poly_channels
from color_cnn import train_cnn_model, cnn_map_color_points
import matplotlib.pyplot as plt

# 1. 读取目标和实际输出csv，只取64x64
def read_csv64(path):
    try:
        with open(path, encoding='utf-8') as f:
            return np.loadtxt(f, delimiter=',', max_rows=64, usecols=range(64))
    except Exception:
        # 文件为空或格式错误时返回全零
        print(f"Error reading {path}, returning zeros.")
        return np.zeros((64, 64))

# 目标sheet（拆分为三组目标）
target = read_csv64('q3_data/target.csv')
target_R = np.zeros((64, 64, 3)); target_R[..., 0] = 220
target_G = np.zeros((64, 64, 3)); target_G[..., 1] = 220
target_B = np.zeros((64, 64, 3)); target_B[..., 2] = 220

# 9个分量sheet
R_R = read_csv64('q3_data/R_R.csv')
R_G = read_csv64('q3_data/R_G.csv')
R_B = read_csv64('q3_data/R_B.csv')
G_R = read_csv64('q3_data/G_R.csv')
G_G = read_csv64('q3_data/G_G.csv')
G_B = read_csv64('q3_data/G_B.csv')
B_R = read_csv64('q3_data/B_R.csv')
B_G = read_csv64('q3_data/B_G.csv')
B_B = read_csv64('q3_data/B_B.csv')

# 组装实际输出
actual_R = np.stack([R_R, R_G, R_B], axis=-1)
actual_G = np.stack([G_R, G_G, G_B], axis=-1)
actual_B = np.stack([B_R, B_G, B_B], axis=-1)

# 合并训练数据
actual_rgb = np.concatenate([actual_R, actual_G, actual_B], axis=0).reshape(-1, 3)
target_rgb = np.concatenate([target_R, target_G, target_B], axis=0).reshape(-1, 3)

# 归一化
actual_rgb_norm = actual_rgb / 255.0
target_rgb_norm = target_rgb / 255.0

# RGB转xy
def rgb_to_xy(rgb):
    # 直接用poly_channels_to_xy和std_rgb_vertices
    rgb = np.asarray(rgb)
    rgb_sum = np.sum(rgb)
    if rgb_sum == 0:
        return np.array([0, 0])
    channels = rgb / rgb_sum
    return poly_channels_to_xy(channels, std_rgb_vertices)

xy_actual = np.array([rgb_to_xy(rgb) for rgb in actual_rgb_norm])
xy_target = np.array([rgb_to_xy(rgb) for rgb in target_rgb_norm])

# xy转为三通道分量
actual_channels = np.array([xy_to_poly_channels(xy, std_rgb_vertices) for xy in xy_actual])
target_channels = np.array([xy_to_poly_channels(xy, std_rgb_vertices) for xy in xy_target])

# 训练参数
EPOCHS = 100
LEARNING_RATE = 1e-2
FEEDBACK_ROUNDS = 0     # 暂时无效参数 
FEEDBACK_BATCH = 32

# 训练CNN模型
cnn_model = train_cnn_model(
    actual_channels, target_channels,
    epochs=EPOCHS, lr=LEARNING_RATE,
    feedback_rounds=FEEDBACK_ROUNDS, feedback_batch=FEEDBACK_BATCH
)

# 校正
mapped_channels = cnn_map_color_points(actual_channels, cnn_model)
mapped_xy = np.array([poly_channels_to_xy(c, std_rgb_vertices) for c in mapped_channels])

# xy转RGB
def xy_to_rgb(xy):
    # 直接用gamut中的反解函数
    rgb = xy_to_poly_channels(xy, std_rgb_vertices)
    rgb = np.clip(rgb, 0, 1)
    return rgb

corrected_rgb = np.array([xy_to_rgb(xy) for xy in mapped_xy])
corrected_rgb_uint8 = (corrected_rgb * 255).clip(0, 255).astype(np.uint8)
corrected_rgb_img = corrected_rgb_uint8.reshape(3, 64, 64, 3)  # 3组(R/G/B目标)

# 保存校正后的数据
np.save('corrected_led_display_data.npy', corrected_rgb_img)
print("校正后的数据已保存。")

# ========== 可视化校正前后6个色块 ==========

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
titles = [
    "beforeR", "beforeG", "beforeB",
    "afterR", "afterG", "afterB"
]
# 校正前
axes[0, 0].imshow(actual_R.astype(np.uint8))
axes[0, 1].imshow(actual_G.astype(np.uint8))
axes[0, 2].imshow(actual_B.astype(np.uint8))
# 校正后
axes[1, 0].imshow(corrected_rgb_img[0])
axes[1, 1].imshow(corrected_rgb_img[1])
axes[1, 2].imshow(corrected_rgb_img[2])
for i, ax in enumerate(axes.flat):
    ax.set_title(titles[i])
    ax.axis('off')
plt.tight_layout()
plt.show()