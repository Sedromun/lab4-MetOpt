import math
from typing import Callable

import numpy as np
from pydantic import BaseModel


class FuncWrapper(BaseModel):
    f: Callable[[float, float], float]
    name: str
    min: float
    logarithmic: bool
    is_infimum: Callable[[float, float], bool]


def near(x: float, y: float, points: list[tuple[float, float]]) -> bool:
    for point in points:
        if abs(x - point[0]) < 0.0001 and abs(y - point[1]) < 0.0001:
            return True
    return False


functions = [
    FuncWrapper(
        f=lambda x, y: -(x ** 2) + y ** 2 + (x ** 4) / 10,
        name="Bubbles",
        min=-2.5,
        logarithmic=True,
        is_infimum=lambda x, y: near(x, y, [(math.sqrt(5), 0), (0, 0), (-math.sqrt(5), 0)])
    ),
    FuncWrapper(
        f=lambda x, y: -(x ** 2) - (y ** 2) + (x ** 4) / 10 + (y ** 4) / 20 + y + 2 * x,
        name="Pig bubbles",
        min=-15.671267003711808,
        logarithmic=True,
        is_infimum=lambda x, y: near(x, y, [(-2.6273503214528366, -3.387640767635553),
                                             (-2.6273503214528366, 2.8740922205040698)])
    ),
    FuncWrapper(
        f=lambda x, y: -np.exp(-(x ** 2) - (y ** 2)),
        name="Bell",
        min=-1,
        logarithmic=False,
        is_infimum=lambda x, y: near(x, y, [(0, 0)])
    ),
    FuncWrapper(
        f=lambda x, y: np.abs(x + y) + 3 * np.abs(y - x),
        name="Euclidean distance",
        min=0,
        logarithmic=False,
        is_infimum=lambda x, y: near(x, y, [(0, 0)])
    ),
    FuncWrapper(
        f=lambda x, y: (1 - x) ** 2 + 100 * (y - x ** 2) ** 2,
        name="Rosenbrock",
        min=0,
        logarithmic=True,
        is_infimum=lambda x, y: near(x, y, [(1, 1)])
    ),
    FuncWrapper(
        f=lambda x, y: x**2 - x*y + y**2 + 9*x - 6*y + 20,
        name="Simple-Dimple",
        min=-1,
        logarithmic=True,
        is_infimum=lambda x, y: near(x, y, [(-4, 1)])
    )
]

#     def foo(x, y):
#         return x ** 3 * y + y ** 2 * x
#
#     print(gessian(start_point, foo))
#
#     x, y = start_point
#
#     print([[6 * x * y, 3 * x ** 2 + 2 * y], [3 * x ** 2 + 2 * y, 2 * x]])
