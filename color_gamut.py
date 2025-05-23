import numpy as np
from matplotlib.path import Path
from scipy.optimize import minimize

# 色域顶点定义（BT.2020 RGB三通道顶点，顺序为R、G、B）
bt2020_vertices = np.array([
    [0.708, 0.292],   # R
    [0.170, 0.797],   # G
    [0.131, 0.046]    # B
])
bt2020_triangle = Path(bt2020_vertices)

# 普通3通道显示器常见的RGB顶点（如sRGB/Rec.709）
std_rgb_vertices = np.array([
    [0.64, 0.33],    # R
    [0.30, 0.60],    # G
    [0.15, 0.06]     # B
])
std_rgb_triangle = Path(std_rgb_vertices)

# 4通道RGVB顶点（可见光范围内，逆时针依次为R、G、V、B）
rgbv_vertices = np.array([
    [0.70, 0.30],    # R
    [0.17, 0.80],    # G
    [0.07, 0.60],    # V
    [0.14, 0.05]     # B
])
rgbv_polygon = Path(rgbv_vertices)

# 5通道RGXCB顶点（可见光范围内，逆时针依次为R、G、X、C、B）
rgbcx_vertices = np.array([
    [0.65, 0.32],    # R
    [0.21, 0.72],    # G
    [0.16, 0.73],    # X
    [0.07, 0.40],    # C
    [0.13, 0.08]     # B
])
rgbcx_polygon = Path(rgbcx_vertices)

def is_in_gamut(xy_point, gamut_polygon):
    """判断色度点是否在色域多边形内"""
    return gamut_polygon.contains_point(xy_point)

def find_closest_boundary_point(xy_point, gamut_polygon):
    """找到多边形边界上最近的点"""
    min_dist = float('inf')
    closest_point = None
    vertices = gamut_polygon.vertices
    n = len(vertices)
    for i in range(n):
        p1, p2 = vertices[i], vertices[(i+1)%n]
        def objective(t):
            pt = p1 + t * (p2 - p1)
            return np.linalg.norm(pt - xy_point)**2
        res = minimize(objective, x0=0.5, bounds=[(0, 1)])
        if res.fun < min_dist:
            min_dist = res.fun
            closest_point = p1 + res.x[0] * (p2 - p1)
    return closest_point

def poly_channels_to_xy(channels, vertices):
    """
    通用多通道到xy的线性映射
    channels: shape (N,), N为通道数，sum=1
    vertices: shape (N,2)
    """
    channels = np.asarray(channels)
    xy = np.dot(channels, vertices)
    return xy

def xy_to_poly_channels(xy, vertices):
    """
    xy到多通道变量的反向映射（最小二乘法，约束channels>=0, sum=1）
    vertices: shape (N,2)
    """
    N = vertices.shape[0]
    def loss(channels):
        return np.linalg.norm(np.dot(channels, vertices) - xy)
    cons = ({'type': 'eq', 'fun': lambda channels: np.sum(channels) - 1},
            {'type': 'ineq', 'fun': lambda channels: channels})
    res = minimize(loss, x0=[1.0/N]*N, constraints=cons)
    return res.x

# 用法示例：
# xy = poly_channels_to_xy([r, g, b], bt2020_vertices)
# xy = poly_channels_to_xy([r, g, b, v], rgbv_vertices)
# xy = poly_channels_to_xy([r, g, b, c, x], rgbcx_vertices)
# 反向：
# channels = xy_to_poly_channels(xy, bt2020_vertices)
# channels = xy_to_poly_channels(xy, rgbv_vertices)
# channels = xy_to_poly_channels(xy, rgbcx_vertices)
