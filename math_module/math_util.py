import numpy as np

from config import epsilon

gessian_epsilon = 0.00001  # with smaller epsilon gt overflow


def derivative_x(x, y, f):
    return (f(epsilon + x, y) - f(x, y)) / epsilon


def derivative_y(x, y, f):
    return (f(x, y + epsilon) - f(x, y)) / epsilon


def gradient(vector, f):
    x, y = vector
    return np.array([derivative_x(x, y, f), derivative_y(x, y, f)])


def second_derivative_x(x, y, f):
    return (f(2 * epsilon + x, y) -
            2 * f(epsilon + x, y) +
            f(x, y)) / (epsilon * epsilon)


def second_derivative_y(x, y, f):
    return (f(x, y + 2 * epsilon) -
            2 * f(x, y + epsilon) +
            f(x, y)) / (epsilon * epsilon)


def gessian(vector, f):
    x, y = vector
    return np.array([[(f(x + gessian_epsilon, y) + f(x - gessian_epsilon, y) - 2 * f(x, y)) / (gessian_epsilon ** 2),
             (f(x + gessian_epsilon, y + gessian_epsilon) - f(x + gessian_epsilon, y) - f(x, y + gessian_epsilon) + f(x, y)) / (gessian_epsilon ** 2)],
            [(f(x + gessian_epsilon, y + gessian_epsilon) - f(x + gessian_epsilon, y) - f(x, y + gessian_epsilon) + f(x, y)) / (gessian_epsilon ** 2),
             (f(x, y + gessian_epsilon) + f(x, y - gessian_epsilon) - 2 * f(x, y)) / (gessian_epsilon ** 2)]])
