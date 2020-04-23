import numpy as np
from scipy.integrate import solve_ivp

def evolve(phi, z0, t, params, n, p, delta, *args):
    def dzdt(t, z):
        return phi(z, t, params, n, p, delta, *args)
    solution = solve_ivp(dzdt, [t[0], t[-1]], z0, method='Radau', t_eval=t)
    return solution.y, solution.success


def evolve_onestep(phi, z, t, params, n, p, delta, *args):
    def dzdt(t, z):
        return phi(z, t, params, n, p, delta, *args)
    solution = solve_ivp(dzdt, [t[0], t[-1]], z, method='Radau', t_eval=t)
    Z = solution.y[:,-1]
    return Z, solution.success


def solve_jac(phi, yhat, t, params, n, p, N, scalar, delta, *args):
    # initial condition J0 = 0
    if scalar:
        y0 = yhat[0]
    else:
        y0 = yhat[:,0]
          
    Z0 = np.zeros(n * (p + 1))
    Z0[0:n] = y0.copy()
    
    Z, suc = evolve(phi, Z0, t, params, n, p, delta, *args)
        
    Y = Z[0:n]
    J = Z[n:]
    Jt = np.hsplit(J,N)
    
    for i in range(N):
        Jt[i] = Jt[i].reshape(p,n).transpose()
        
    return Y, Jt


if __name__ == '__main__':
    import time
    from data_import import importing
    from pbm import phi
    
    k0 = np.array([1e-1, 0.8, 0.15])
    length, volume, number, Y0, mu, sigma, t, n, N, p, Q = importing(k0)
    args = [length, mu, sigma]
    tic = time.time()
    Y, Jac = solve_jac(phi, number, t, k0, n, p, N, False, 1e-8, *args)
    toc = time.time() - tic
    print('solving ODE took %5.2f seconds' % toc)