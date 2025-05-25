import numpy as np
from color_gamut import std_rgb_vertices, poly_channels_to_xy, xy_to_poly_channels
# from color_cnn import train_cnn_model, cnn_map_color_points
from color_cnn_q3fix import train_cnn_model, cnn_map_color_points  # 用新模型
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

# 训练参数
EPOCHS = 1000
LEARNING_RATE = 1e-2
FEEDBACK_ROUNDS = 0     # 暂时无效参数 
FEEDBACK_BATCH = 32

print("Training CNN model...")

# 训练CNN模型
cnn_model = train_cnn_model(
    actual_rgb_norm, target_rgb_norm,
    epochs=EPOCHS, lr=LEARNING_RATE,
    feedback_rounds=FEEDBACK_ROUNDS, feedback_batch=FEEDBACK_BATCH
)

# 校正（直接映射RGB归一化数据）
mapped_rgb_norm = cnn_map_color_points(actual_rgb_norm, cnn_model)
mapped_rgb_norm = np.nan_to_num(mapped_rgb_norm, nan=0.0, posinf=1.0, neginf=0.0)
corrected_rgb_uint8 = (mapped_rgb_norm * 255).clip(0, 255).astype(np.uint8)
corrected_rgb_img = corrected_rgb_uint8.reshape(3, 64, 64, 3)  # 3组(R/G/B目标)

# 保存校正后的数据
np.save('corrected_led_display_data.npy', corrected_rgb_img)
print("校正后的数据已保存。")

# ========== 生成12个色块（校正前/后主色块、黄/青/品红、渐变） ==========

def make_color_block(color, shape=(64, 64, 3)):
    block = np.zeros(shape)
    block[..., 0] = color[0]
    block[..., 1] = color[1]
    block[..., 2] = color[2]
    return block

def correct_block(block):
    flat = block.reshape(-1, 3) / 255.0
    mapped = cnn_map_color_points(flat, cnn_model)
    mapped = np.nan_to_num(mapped, nan=0.0, posinf=1.0, neginf=0.0)
    rgb_corr = (mapped * 255).clip(0, 255).astype(np.uint8)
    return rgb_corr.reshape(block.shape)

# 主色块
main_blocks_before = [
    actual_R.astype(np.uint8),
    actual_G.astype(np.uint8),
    actual_B.astype(np.uint8)
]
main_blocks_after = [
    corrected_rgb_img[0],
    corrected_rgb_img[1],
    corrected_rgb_img[2]
]

# 黄、青、品红
block_yellow = make_color_block([220, 220, 0])
block_cyan = make_color_block([0, 220, 220])
block_magenta = make_color_block([220, 0, 220])
block_yellow_corr = correct_block(block_yellow)
block_cyan_corr = correct_block(block_cyan)
block_magenta_corr = correct_block(block_magenta)

main_blocks_before += [
    block_yellow.astype(np.uint8),
    block_cyan.astype(np.uint8),
    block_magenta.astype(np.uint8)
]
main_blocks_after += [
    block_yellow_corr,
    block_cyan_corr,
    block_magenta_corr
]

# 渐变色块
grad_rb = np.zeros((64, 64, 3))
for i in range(64):
    t = i / 63
    grad_rb[:, i, 0] = 220 * (1 - t)
    grad_rb[:, i, 2] = 220 * t
grad_gray = np.zeros((64, 64, 3))
for i in range(64):
    v = int(220 * i / 63)
    grad_gray[:, i, :] = v

grad_rb_corr = correct_block(grad_rb)
grad_gray_corr = correct_block(grad_gray)

main_blocks_before += [
    grad_rb.astype(np.uint8),
    grad_gray.astype(np.uint8)
]
main_blocks_after += [
    grad_rb_corr,
    grad_gray_corr
]

# ========== 合并12个色块到一起展示 ==========

titles_before = [
    "Before - Red", "Before - Green", "Before - Blue",
    "Before - Yellow", "Before - Cyan", "Before - Magenta",
    "Before - Red-Blue Gradient", "Before - Grayscale Gradient"
]
titles_after = [
    "After - Red", "After - Green", "After - Blue",
    "After - Yellow", "After - Cyan", "After - Magenta",
    "After - Red-Blue Gradient", "After - Grayscale Gradient"
]

# 2. 合并前后，共12个
all_blocks = main_blocks_before + main_blocks_after
all_titles = titles_before + titles_after

fig, axes = plt.subplots(2, 8, figsize=(24, 7))
for i, ax in enumerate(axes.flat):
    if i < len(all_blocks):
        ax.imshow(all_blocks[i])
        ax.set_title(all_titles[i])
    else:
        ax.axis('off')
    ax.axis('off')
plt.tight_layout()
plt.show()
