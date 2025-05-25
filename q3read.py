import numpy as np
import matplotlib.pyplot as plt

# 读取校正后的数据
corrected_rgb_img = np.load('corrected_led_display_data.npy')  # shape: (3, 64, 64, 3)

# ========== 生成未校正的主色块、黄/青/品红、渐变 ==========

def make_color_block(color, shape=(64, 64, 3)):
    block = np.zeros(shape)
    block[..., 0] = color[0]
    block[..., 1] = color[1]
    block[..., 2] = color[2]
    return block

# 主色块（未校正）
actual_R = make_color_block([220, 0, 0])
actual_G = make_color_block([0, 220, 0])
actual_B = make_color_block([0, 0, 220])

# 黄、青、品红（未校正）
block_yellow = make_color_block([220, 220, 0])
block_cyan = make_color_block([0, 220, 220])
block_magenta = make_color_block([220, 0, 220])

# 渐变色块（未校正）
grad_rb = np.zeros((64, 64, 3))
for i in range(64):
    t = i / 63
    grad_rb[:, i, 0] = 220 * (1 - t)
    grad_rb[:, i, 2] = 220 * t
grad_gray = np.zeros((64, 64, 3))
for i in range(64):
    v = int(220 * i / 63)
    grad_gray[:, i, :] = v

main_blocks_before = [
    actual_R.astype(np.uint8),
    actual_G.astype(np.uint8),
    actual_B.astype(np.uint8),
    block_yellow.astype(np.uint8),
    block_cyan.astype(np.uint8),
    block_magenta.astype(np.uint8),
    grad_rb.astype(np.uint8),
    grad_gray.astype(np.uint8)
]
main_blocks_after = [
    corrected_rgb_img[0],
    corrected_rgb_img[1],
    corrected_rgb_img[2],
    # 其余色块校正后无法直接从npy获得，跳过
    np.zeros((64, 64, 3), dtype=np.uint8),  # 占位
    np.zeros((64, 64, 3), dtype=np.uint8),  # 占位
    np.zeros((64, 64, 3), dtype=np.uint8),  # 占位
    np.zeros((64, 64, 3), dtype=np.uint8),  # 占位
    np.zeros((64, 64, 3), dtype=np.uint8)   # 占位
]

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

# 只展示主色块校正前后对比（红绿蓝）
fig2, axes2 = plt.subplots(2, 3, figsize=(12, 6))
for i in range(3):
    axes2[0, i].imshow(main_blocks_before[i])
    axes2[0, i].set_title(titles_before[i])
    axes2[0, i].axis('off')
    axes2[1, i].imshow(main_blocks_after[i])
    axes2[1, i].set_title(titles_after[i])
    axes2[1, i].axis('off')
plt.tight_layout()
plt.show()
