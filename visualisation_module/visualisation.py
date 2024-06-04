import math
from math_module.functions import FuncWrapper

import matplotlib.pyplot as plt
import numpy as np


def get_grid(function, grid_step, radius):
    samples = np.arange(-radius, radius, grid_step)
    x, y = np.meshgrid(samples, samples)
    return x, y, function(x, y)


def draw_chart(
        function: FuncWrapper,
        point: tuple[float, float],
        title: str = "",
        grid_step: float = 0.05,
        radius: float = 4
):
    point_x, point_y, point_z = point[0], point[1], function.f(point[0], point[1])
    grid_x, grid_y, grid_z = get_grid(function.f, grid_step, radius)
    plt.rcParams.update({
        'figure.figsize': (4, 4),
        'figure.dpi': 200,
        'xtick.labelsize': 4,
        'ytick.labelsize': 4
    })
    ax = plt.figure().add_subplot(111, projection='3d')
    ax.scatter(point_x, point_y, point_z, color='red')
    ax.plot_surface(grid_x, grid_y, grid_z, rstride=5, cstride=5, alpha=0.7)
    plt.title(f'Method: {title}\nFunction: {function.name}')
    plt.show(block=True)


def draw_graphic(
        points: list[tuple[float, float]],
        function: FuncWrapper,
        title: str = ""
):
    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    n = len(points)
    ax.plot([i for i in range(n)],
            [math.log(function.f(p[0], p[1]) - function.f(points[n - 1][0], points[n - 1][1]) + 1) for p in
             points])  # Plot some data on the axes.
    ax.set_xlabel('Iterations')
    ax.set_ylabel('log[ f(P_i) - f(P_n) + 1 ]')
    plt.title(f'Method: {title}\nFunction: {function.name}')
    plt.show()


def dist(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def reducer():
    summ = 0

    def foo(term: float) -> float:
        nonlocal summ
        summ += term
        return summ

    return foo


def draw_graphic_2(
        points: list[tuple[float, float]],
        function: FuncWrapper,
        title: str = ""
):
    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    n = len(points)
    summer = reducer()
    ax.plot(
        [0 if i == 0 else summer(dist(points[i - 1], points[i])) for i in range(n)],
        [function.f(p[0], p[1]) - function.f(points[n - 1][0], points[n - 1][1]) for p in points],
        marker='o',
    )
    ax.set_xlabel('Distance from start point')
    ax.set_ylabel('Value in point')
    plt.title(f'Method: {title}\nFunction: {function.name}')
    plt.show()


def draw_isolines(
        points: list[tuple[float, float]],
        function: FuncWrapper,
        title: str = "",
        grid_step: float = 0.05,
        radius: float = 8
):
    points = np.array(points)
    plt.figure(figsize=(8, 6))
    # min_x = points[-1][0]
    # min_y = points[-1][1]
    grid_x, grid_y, grid_z = get_grid(
        lambda x, y: function.f(x, y) - function.min + 1.01,
        grid_step,
        radius)
    plt.contour(grid_x, grid_y, grid_z,
                levels=(np.logspace(-0.5, 3.5, radius * 3) if function.logarithmic else None),
                cmap='gray'
                )
    plt.plot(points[:, 0], points[:, 1], marker='o', color='r', markersize=3, linestyle='-', linewidth=1)

    # for Nelder Mid
    # for i in range(2, len(points)):
    #     if i % 4 != 5:
    #         plt.plot([points[i-2][0], points[i-1][0], points[i][0], points[i-2][0]], [points[i-2][1], points[i-1][1], points[i][1], points[i-2][1]],
    #                  marker='o', color='b', markersize=3, linestyle='-', linewidth=1)

    plt.plot(points[0][0], points[0][1], marker='o', color='g', markersize=10)  # Стартовая точка
    plt.title(f'Method: {title}\nFunction: {function.name}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
