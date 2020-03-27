import numpy as np
from scipy.integrate import solve_ivp

from discretize import discretize
from ode import breakage
from phi_orgn import phi_breakage

def evolve(odes, z0, dbs, t, n, p, delta=1e-8):
    def dzdt(t, z):
        return phi_breakage(odes, z, dbs, n, p, delta)
    solution = solve_ivp(dzdt, [t[0],t[-1]], z0, method='Radau', t_eval=t)
    return solution.y, solution.success


def evolve_onestep(odes, z, dbs, t, n, p, delta):
    def dxdt(t, x):
        return phi_breakage(odes, x, dbs, n, p, delta)
    solution = solve_ivp(dxdt, [t[0],t[-1]], z, method = 'Radau', t_eval = t)
    Z = solution.y[:,-1]
    return Z, solution.success

def solve_jac(odes, yhat, dbs, t, n, p, N, scalar, delta):
    print('Solving ODE...')
    # initial condition J0 = 0
    if scalar:
        y0 = yhat[0]
    else:
        y0 = yhat[:,0]
          
    Z0 = np.zeros(n * (p + 1))
    Z0[0:n] = y0.copy()
    
    Z, suc = evolve(odes, Z0, dbs, t, n, p, delta=1e-8)
        
    Y = Z[0:n]
    J = Z[n:]
    Jt = np.hsplit(J,N)
    
    for i in range(N):
        Jt[i] = Jt[i].reshape(p,n).transpose()
        
    return Y, Jt


if __name__ == '__main__':
    import time
    from data_import import importing
    
    k0 = np.array([1e-7, 0.8, 0.15])
    length, volume, number, Y0, mu, sigma, t, n, N, p, Q = importing(k0)
    arguments = [mu, sigma]
    tic = time.time()
    dbs = discretize(length, n, p, k0, 1e-8, *arguments)
    duration = time.time() - tic
    print('discretization takes %f seconds' % duration)
    tic = time.time()
    Y, Jac = solve_jac(breakage, number, dbs, t, n, p, N, False, delta=1e-8)
    duration = time.time() - tic
    print('solving ODE takes %f seconds' % duration)