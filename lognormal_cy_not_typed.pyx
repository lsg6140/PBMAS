# cython: language_level=3

import numpy as np
from libc.math cimport exp, log, sqrt, erfc

def lnpdf(x, m, sg):
    num = exp(-(log(x) - m)**2 / (2 * sg**2))
    den = x * sg * sqrt(2 * np.pi)
    return num / den

def lognorm_b(x, y, m, sg):
    assert sg > 0, "sigma must be larger than 0"
   
    num = lnpdf(x, m, sg)
    den = erfc(-(log(y) - m) / (sqrt(2) * sg))/2

    if den == 0:
        den = np.finfo(float).eps

    return (y / x)**3 * num / den

def breakagefunc(x, y, k, *args):
    mu = args[0]
    sigma = args[1]
    res = k[1] * lognorm_b(x, y, mu[0], sigma[0])\
        + k[2] * lognorm_b(x, y, mu[1], sigma[1])\
        + (1 - k[1] - k[2]) * lognorm_b(x, y, mu[2], sigma[2])
    return res

def selectionfunc(y, k, *args):
    return k[0] * y**3