# вычисляет минимум на отрезке [a, b] с точностью epsilon для унимодальных функций
from config import gradient_learning_rate


def calc_learning_rate(f, x, direction, max_learning_rate):
    def g(t: float):
        vector = x - t * direction
        return f(vector[0], vector[1])

    return dichotomy_method(g, 0, max_learning_rate)


def dichotomy_method(f, a, b):
    epsilon = 0.001
    if f(a) * f(b) > 0:
        return b

    while (b - a) / 2 > epsilon:
        c = (a + b) / 2
        if f(c) == 0:
            return c
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2
