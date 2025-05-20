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

def is_in_gamut(xy_point, gamut_triangle):
    """判断色度点是否在色域内"""
    return gamut_triangle.contains_point(xy_point)

def find_closest_boundary_point(xy_point, gamut_triangle):
    """找到目标色域边界上最近的点"""
    min_dist = float('inf')
    closest_point = None
    vertices = gamut_triangle.vertices
    for i in range(3):
        p1, p2 = vertices[i], vertices[(i+1)%3]
        def objective(t):
            pt = p1 + t * (p2 - p1)
            return np.linalg.norm(pt - xy_point)**2
        res = minimize(objective, x0=0.5, bounds=[(0, 1)])
        if res.fun < min_dist:
            min_dist = res.fun
            closest_point = p1 + res.x[0] * (p2 - p1)
    return closest_point
