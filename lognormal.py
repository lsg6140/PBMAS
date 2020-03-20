from scipy.special import erfc

def lnpdf(x, mu, sigma):
    num = np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))
    den = x * sigma * np.sqrt(2 * np.pi)
    return num / den

def lognorm_b(x, l, mu, sigma):
    assert sigma > 0, "sigma must be larger than 0"
   
    num = lnpdf(x, mu, sigma)
    den = erfc(-(np.log(l) - mu) / (np.sqrt(2) * sigma))/2

    # In case 'l' is too small compared to 'mu',
    # 'den' can be numerically zero 
    # if it is smaller than the machine precision epsilon 
    # which is not correct theoretically
    if den == 0:
        den = np.finfo(float).eps
    # convert volume to number
    return (l/x)**3 * num / den

def breakagefunc(x, l, k, mu_arr, sigma_arr):
    bf=k[1] * lognorm_b(x,l,mu_arr[0],sigma_arr[0])\
      + k[2] * lognorm_b(x,l,mu_arr[1],sigma_arr[1])\
      + (1-k[1]-k[2])*lognorm_b(x,l,mu_arr[2],sigma_arr[2])
    return bf

def selectionfunc(l, k):
    return k[0] * l**3