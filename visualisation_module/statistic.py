import time
from typing import Callable

from random import randint as rand

from tabulate import tabulate

from math_module.functions import FuncWrapper, functions


def stat_method(
        method: Callable[[Callable[[float, float], float], tuple[float, float]], tuple[float, float]],
        func: FuncWrapper,
):
    attempts, success, semi_success, errors, call = 1000, 0, 0, 0, 0

    elapsed_time = 0

    def wrap(x, y):
        nonlocal call
        call += 1
        return func.f(x, y)

    for _ in range(attempts):
        start = (rand(-8, 8), rand(-8, 8))
        try:
            start_time = time.time()
            x, y = method(wrap, start)
            end_time = time.time()
        except Exception:
            errors += 1
            # print(f"ERROR, func={func.name}")
        else:
            elapsed_time += end_time - start_time
            if abs(func.f(x, y) - func.min) < 0.001:
                success += 1
            elif func.is_infimum(x, y):
                semi_success += 1

    return attempts, success, semi_success, errors, call, elapsed_time


headers = ["Function", "Attempts", "Success", "Semi-success", "Incorrect", "Errors", "Average func calls",
           "Average time"]


def sub_stat(method, name):
    results = []
    t = 0
    for func in functions:
        a, s, ss, e, c, t = stat_method(method, func)
        results.append((func.name, a, s + ss, ss, a - s - ss, e, c / a, (str(round(1000 * t / (a - e), 2)) + " ms" if a > e else "-")))
    print(f"{name}; time = {1000*t:.2f} ms")
    print(tabulate(results, headers=headers))
    print("\n\n\n")
