# cython: language_level=3

import numpy as np
from libc.math cimport exp, log, sqrt, erfc

cdef double lnpdf(double x, double m, double sg):
    cdef double pi = 3.141592653589793115997963468544185161590576171875
    cdef double num = exp(-(log(x) - m) ** 2 / (2 * sg**2))
    cdef double den = x * sg * sqrt(2 * pi)
    return num / den

cdef double lognorm_b(double x, double y, double m, double sg):
    assert sg > 0, "sigma must be larger than 0"   
    m += sg**2
    cdef double num = lnpdf(x, m, sg)
    cdef double den = erfc(-(log(y) - m) / (sqrt(2) * sg)) / 2
    if den == 0:
        den = np.finfo(float).eps
    return (y / x)**3 * num / den

cpdef double breakagefunc(double x, double y, double[:] k, args):
    cdef double[:] mu = args[0]
    cdef double[:] sigma = args[1]
    cdef double res = k[1] * lognorm_b(x, y, mu[0], sigma[0])\
                    + k[2] * lognorm_b(x, y, mu[1], sigma[1])\
                    + (1 - k[1] - k[2]) * lognorm_b(x, y, mu[2], sigma[2])
    return res

cpdef double selectionfunc(double y, double[:] k, args):
    return k[0] * y**3