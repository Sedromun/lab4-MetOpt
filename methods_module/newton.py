import numpy as np

from config import gradient_learning_rate, newton_learning_rate
from methods_module.gradient import FunctionNonConvergence
from math_module.math_util import *


def newton(func, start_point=None, calc_learning_rate=(lambda x, y, z, t: newton_learning_rate)):
    points = [np.array(start_point) if (start_point is not None) else np.array([0, 0])]
    while True:
        last = points[-1]
        grad = gradient(last, func)
        hessian = gessian(last, func)
        if np.linalg.norm(grad) < epsilon:
            break

        direction = np.linalg.inv(hessian).dot(grad)

        new_point = last - calc_learning_rate(func, last, direction, newton_learning_rate) * direction

        points.append(new_point)
        if len(points) > 1000 or abs(points[-1][0]) > 10000000 or abs(points[-1][1]) > 10000000:
            raise FunctionNonConvergence()
    return points
