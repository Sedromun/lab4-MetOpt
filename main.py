from typing import Callable

from math_module.functions import functions
from methods_module.gradient import FunctionNonConvergence, gradient_descent
from methods_module.my_bfgs import my_bfgs
from methods_module.newton import newton
from visualisation_module.statistic import sub_stat
from visualisation_module.visualisation import *
from methods_module.scipy_methods import *
from methods_module.d1_methods import *
from methods_module.coordinate_descent import *
from random import randint as rand
from math_module.functions import functions
from tabulate import tabulate
from methods_module.my_bfgs import my_bfgs
import time

points = []


# function from habr
def rosen(x, y):
    return 100.0 * (y - x ** 2.0) ** 2.0 + (1 - x) ** 2.0


def rosen_jac(x, y):
    m1 = -400.0 * x * (y - x ** 2.0) - 2 * (1 - x)
    m2 = 200 * (y - x ** 2.0)
    return [m1, m2]


def rosen_hess(x, y):
    m11 = 1200.0 * x ** 2.0 - 400 * y + 2
    m12 = 0
    m21 = 0
    m22 = 200 * y
    return [[m11, m12],
            [m21, m22]]


def logger(f: Callable[[float, float], float]) -> Callable[[float, float], float]:
    def foo(x: float, y: float) -> float:
        points.append((x, y))
        return f(x, y)

    return foo


def process_gradient_descent(func, start):
    try:
        x, y = gradient_descent(logger(func.f), start_point=start)
    except FunctionNonConvergence:
        print('ERROR start point: ', start)
    else:
        print("GRADIENT DESCENT: ", x, y, " Value :=", func.f(x, y))
        draw(points, func, x, y, title="Gradient Descent")


def process_d1_search_gradient(func, start):
    try:
        x, y = gradient_descent(logger(func.f), start_point=start, calc_learning_rate=calc_learning_rate)
    except FunctionNonConvergence:
        print('ERROR start point: ', start)
    else:
        print("GRADIENT DESCENT WITH D1 OPTIMIZATION: ", x, y, " Value :=", func.f(x, y))
        draw(points, func, x, y, title="Gradient Descent with D1 optimization")


def process_coordinate_descent(func, start):
    x, y, p = coordinate_descent(logger(func.f), start)
    print("COORDINATE DESCENT: ", x, y, " Value :=", func.f(x, y))
    draw(p, func, x, y, title="Coordinate Descent")


def process_nelder_mead(func, start):
    x, y = nelder_mead(logger(func.f), start)
    print("NELDER-MEAD: ", x, y, " Value :=", func.f(x, y))
    draw(points, func, x, y, title="Nelder-Mead")


def process_newton(func, start):
    try:
        newton_points = newton(func.f, start_point=start, calc_learning_rate=(lambda a, b, c, d: 1))
        x, y = newton_points[-1]
    except FunctionNonConvergence:
        print('ERROR start point: ', start)
    except Exception as e:
        print('ERROR start point: ', start, " Error:", e)
    else:
        print("NEWTON's METHOD: ", x, y, " Value :=", func.f(x, y))
        draw(newton_points, func, x, y, title="Newton's Method")


def process_d1_search_newton(func, start):
    try:
        newton_points = newton(func.f, start_point=start, calc_learning_rate=calc_learning_rate)
        x, y = newton_points[-1]
        print(newton_points)
    except FunctionNonConvergence:
        print('ERROR start point: ', start)
    except Exception as e:
        print('ERROR start point: ', start, " Error:", e)
        raise e
    else:
        print("NEWTON's METHOD WITH D1 OPTIMIZATION: ", x, y, " Value :=", func.f(x, y))
        draw(newton_points, func, x, y, title="Newton's Method with D1 optimization")


def process_newton_cg(func, start):
    x, y = newton_cg(logger(func.f), start)
    print("NEWTON-CG: ", x, y, " Value :=", func.f(x, y))
    draw(points, func, x, y, title="Newton-CG")


def process_BFSG(func, start):
    x, y = BFSG(logger(func.f), start)
    print("BFSG: ", x, y, " Value :=", func.f(x, y))
    draw(points, func, x, y, title="BFSG")


def process_my_BFSG(func, start):
    x, y = my_bfgs(logger(func.f), start)
    print("BFSG: ", x, y, " Value :=", func.f(x, y))
    draw(points, func, x, y, title="my BFSG")


def draw(dots, func, x, y, title: str = ""):
    draw_graphic(dots, func, title=title)
    draw_graphic_2(dots, func, title=title)
    draw_isolines(dots, func, title=title)
    draw_chart(func, (x, y), title=title)


def stat():
    sub_stat(gradient_descent, "GRADIENT DESCENT")
    sub_stat(lambda f, p: gradient_descent(f, p, calc_learning_rate), "GRADIENT DESCENT D1")
    sub_stat(coordinate_descent, "COORDINATE DESCENT")
    sub_stat(nelder_mead, "NELDER MEAD")

    sub_stat(newton, "NEWTON")
    sub_stat(lambda f, p: newton(f, p, calc_learning_rate), "NEWTON WITH D1 OPTIMIZATION")
    sub_stat(newton_cg, "NEWTON-CG")
    sub_stat(my_bfgs, "JEKA's BFSG")
    sub_stat(BFSG, "BFSG")


def run(func, st_point):
    process_gradient_descent(func, st_point)
    process_d1_search_gradient(func, st_point)
    process_coordinate_descent(func, st_point)
    process_nelder_mead(func, start_point)

    # process_newton(func, start_point)
    # process_d1_search_newton(func, st_point)
    process_newton_cg(func, st_point)
    process_BFSG(func, st_point)
    process_my_BFSG(func, st_point)


if __name__ == '__main__':
    start_point = (rand(-8, 8), rand(-8, 8))
    # stat()

    process_coordinate_descent(functions[4], (6, -5))

    # run(functions[0], start_point)  # TODO
