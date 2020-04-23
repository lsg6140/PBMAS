from discretize_parallel_critical import discretize
from phi_cy_parallel import phi_breakage

def phi(z, t, params, n, p, delta, *args):
    dbs = discretize(args[0], n, p, params, delta, args[1], args[2], args[3])
    
    return phi_breakage(z, dbs, n, p, delta)


if __name__ == '__main__':
    import numpy as np
    from data_import import importing
    from pbm import phi
    
    k0 = np.array([1e-1, 0.8, 0.15, 3.0, 50.0])
    length, volume, number, Y0, mu, sigma, t, n, N, p, Q = importing(k0)
    args = [length, mu, sigma]
    
    y0 = number[:,0]
        
    Z0 = np.zeros(n * (p + 1))
    Z0[0:n] = y0.copy()   
    
    res = phi(Z0, t[0], k0, n, p, 1e-8, *args)