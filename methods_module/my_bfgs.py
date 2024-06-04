import numpy as np
import numpy.linalg as ln
import scipy
import warnings
from math_module.math_util import gradient

from math_module.math_util import gessian

from config import epsilon


def my_bfgs(f1, x0):
    warnings.filterwarnings("ignore")
    def fprime(x):
        a1, b1 = gradient(x, f1)
        return np.array([a1, b1])

    def f(x):
        return f1(x[0], x[1])

    gfk = fprime(x0)
    N = len(x0)
    I = np.eye(N, dtype=int)
    k, l = gessian(x0, f1)
    Hk = np.linalg.inv(np.array([k, l]))
    xk = x0

    while ln.norm(gfk) > epsilon:
        # pk - direction of search

        pk = -np.dot(Hk, gfk)

        # Line search constants for the Wolfe conditions.
        # Repeating the line search
        # line_search returns not only alpha
        # but only this value is interesting for us

        line_search = scipy.optimize.line_search(f, fprime, xk, pk, maxiter=10000000)
        alpha_k = line_search[0]
        xkp1 = xk + alpha_k * pk
        sk = xkp1 - xk
        xk = xkp1

        gfkp1 = fprime(xkp1)
        yk = gfkp1 - gfk
        gfk = gfkp1

        ro = 1.0 / (np.dot(yk, sk))
        A1 = I - ro * sk[:, np.newaxis] * yk[np.newaxis, :]
        A2 = I - ro * yk[:, np.newaxis] * sk[np.newaxis, :]
        Hk = np.dot(A1, np.dot(Hk, A2)) + (ro * sk[:, np.newaxis] *
                                           sk[np.newaxis, :])

    return xk[0], xk[1]
