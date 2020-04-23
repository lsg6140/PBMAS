from scipy.special import erfc
import numpy as np

def lnpdf(x, m, sg):
    num = np.exp(-(np.log(x) - m)**2 / (2 * sg**2))
    den = x * sg * np.sqrt(2 * np.pi)
    return num / den

def lognorm_b(x, y, m, sg):
    assert sg > 0, "sigma must be larger than 0"
    m += sg**2
   
    num = lnpdf(x, m, sg)
    den = erfc(-(np.log(y) - m) / (np.sqrt(2) * sg))/2

    # In case 'y' is too small compared to 'mu',
    # 'den' can be numerically zero 
    # if it is smaller than the machine precision epsilon 
    # which is not correct theoretically
    if den == 0:
        den = np.finfo(float).eps
    # convert volume to number
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
