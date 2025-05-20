import numpy as np

def xyY_to_XYZ(xyY):
    """将 xyY 转换为 XYZ"""
    x, y, Y = xyY
    if y == 0:
        return np.array([0, 0, 0])
    X = (x / y) * Y
    Z = ((1 - x - y) / y) * Y
    return np.array([X, Y, Z])

def xy_to_XYZ(xy, Y=1.0):
    """仅xy转XYZ，Y可设为1"""
    x, y = xy
    if y == 0:
        return np.array([0, 0, 0])
    X = (x / y) * Y
    Z = ((1 - x - y) / y) * Y
    return np.array([X, Y, Z])

def deltaE_xy(xy1, xy2):
    """以xy为输入，计算近似deltaE（在Y=1的XYZ空间欧氏距离）"""
    XYZ1 = xy_to_XYZ(xy1)
    XYZ2 = xy_to_XYZ(xy2)
    return np.linalg.norm(XYZ1 - XYZ2)

def loss_function(original_xy, mapped_xy):
    """计算色度损失（欧氏距离）"""
    return np.linalg.norm(original_xy - mapped_xy)

def generate_points_in_triangle(vertices, num_points):
    """在给定三角形内生成均匀分布的随机点"""
    points = []
    for _ in range(num_points):
        r1, r2 = np.random.rand(2)
        sqrt_r3 = np.sqrt(np.random.rand())
        a = 1 - sqrt_r3
        b = r1 * sqrt_r3
        c = (1 - r1) * sqrt_r3
        point = a * vertices[0] + b * vertices[1] + c * vertices[2]
        points.append(point)
    return np.array(points)

def generate_points_outside_triangle(vertices, num_points):
    """生成在三角形外的随机点"""
    points = []
    for _ in range(num_points):
        offset = np.random.uniform(-0.1, 0.1, size=2)
        base_point = vertices[np.random.randint(0, 3)]
        point = base_point + offset
        points.append(point)
    return np.array(points)
