from config import epsilon


def coordinate_descent(f, start_point):
    points = []

    step = 1
    stopped = 0
    x = start_point[0]
    y = start_point[1]
    current = f(x, y)

    while step > epsilon:
        points.append((x, y))

        current_plus = f(x + step, y)
        current_minus = f(x - step, y)
        if current > current_plus:
            current = current_plus
            x += step
            stopped = 0
        elif current > f(x - step, y):
            current = current_minus
            x -= step
            stopped = 0
        else:
            stopped += 1

        if stopped >= 2:
            step /= 2
            stopped = 0
            if step < epsilon:
                break

        points.append((x, y))

        current_plus = f(x, y + step)
        current_minus = f(x, y - step)
        if current > current_plus:
            current = current_plus
            y += step
            stopped = 0
        elif current > current_minus:
            current = current_minus
            y -= step
            stopped = 0
        else:
            stopped += 1

        if stopped >= 2:
            step /= 2
            stopped = 0

    return x, y, points
