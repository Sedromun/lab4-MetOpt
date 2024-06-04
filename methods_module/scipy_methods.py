import scipy

from math_module.math_util import gradient

from math_module.math_util import gessian


def nelder_mead(f, start_point):
    def n_m_f(x):
        return f(x[0], x[1])

    res = scipy.optimize.minimize(n_m_f, start_point, method='Nelder-Mead')
    return res.x[0], res.x[1]


def newton_cg(f, start_point):
    def scipy_f(x):
        return f(x[0], x[1])

    def scipy_j(x):
        a, b = gradient(x, f)
        return [a, b]

    def scipy_h(x):
        a, b = gessian(x, f)
        return [a, b]

    res = scipy.optimize.minimize(scipy_f, start_point, method='Newton-CG', jac=scipy_j, hess=scipy_h)
    return res.x[0], res.x[1]


def BFSG(f, start_point):
    def scipy_f(x):
        return f(x[0], x[1])

    res = scipy.optimize.minimize(scipy_f, start_point, method='BFGS')
    return res.x[0], res.x[1]
