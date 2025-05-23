import numpy as np
from matplotlib.path import Path
from scipy.optimize import minimize

# 色域顶点定义
bt2020_vertices = np.array([
    [0.708, 0.292],
    [0.170, 0.797],
    [0.131, 0.046]
])
bt2020_triangle = Path(bt2020_vertices)

std_rgb_vertices = np.array([
    [0.64, 0.33],
    [0.30, 0.60],
    [0.15, 0.06]
])
std_rgb_triangle = Path(std_rgb_vertices)

# 4通道RGBV顶点（示例坐标，可根据实际修改）
rgbv_vertices = np.array([
    [0.68, 0.32],   # R
    [0.21, 0.71],   # G
    [0.10, 0.60],   # V
    [0.15, 0.06]    # B
])
rgbv_polygon = Path(rgbv_vertices)

# 5通道RGBCX顶点（示例坐标，可根据实际修改）
rgbcx_vertices = np.array([
    [0.60, 0.34],   # R
    [0.31, 0.60],   # G
    [0.18, 0.74],   # X
    [0.10, 0.40],   # C
    [0.10, 0.10]    # B
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
