import numpy as np

from config import gradient_learning_rate, epsilon

from math_module.math_util import gradient


class FunctionNonConvergence(Exception):
    def __init__(self):
        super(FunctionNonConvergence, self)


def gradient_descent(func, start_point=None, calc_learning_rate=(lambda x, y, z, t: gradient_learning_rate)):
    points = [np.array(start_point) if (start_point is not None) else np.array([0, 0])]
    while len(points) < 2 or np.linalg.norm(points[-1] - points[-2]) > epsilon:
        grad = gradient(points[-1], func)
        points.append(points[-1] -
                      calc_learning_rate(func, points[-1], grad, gradient_learning_rate) *
                      grad)
        if len(points) > 1000 or abs(points[-1][0]) > 10000000 or abs(points[-1][1]) > 10000000:
            raise FunctionNonConvergence()
    return points[-1]
