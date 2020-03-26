import numpy as np
from scipy.integrate import solve_ivp
from joblib import Memory

cachedir = './cachedir'
memory = Memory(cachedir)

from discretize import breakage_discretize, selection_discretize
from pbm import breakage

def evolve(ode, z0, dbs, t, n, p, delta=1e-8):
    def dzdt(t, z):
        return phi_breakage(ode, z, dbs, n, p, delta)
    solution = solve_ivp(dzdt, [t[0],t[-1]], z0, method='Radau', t_eval=t)
    return solution.y, solution.success


def evolve_onestep(ode, z, dbs, t, n, p, delta):
    def dxdt(t, x):
        return phi_breakage(ode, x, dbs, n, p, delta)
    solution = solve_ivp(dxdt, [t[0],t[-1]], z, method = 'Radau', t_eval = t)
    Z = solution.y[:,-1]
    return Z, solution.success


def phi_breakage(breakage, z, dbs, n, p, delta):
    # dbs: discretized breakage and selection functions
    z = z.astype(np.float)
    y = z[0:n]
    J = z[n:].reshape((p, n)).transpose()
    phiz = np.empty(n * (p+1))
    dfdy = np.empty((n, n))
    dfdk = np.empty((p, n))
    
    for i in range(n):
        yr = y.copy()
        
        yl = y.copy()
        yr[i] += delta
        yl[i] -= delta
        dfdy[i] = (breakage(yr, dbs[0], dbs[1]) - \
                   breakage(yl, dbs[0], dbs[1])) / (2 * delta)
    dfdy = dfdy.transpose()
    
    for i in range(p):
        dfdk[i] = (breakage(y, dbs[2][i], dbs[3][i]) - \
                   breakage(y, dbs[4][i], dbs[5][i])) / (2 * delta)
    dfdk = dfdk.transpose()
    
    dJdt = dfdy @ J + dfdk
    phiz[0:n] = breakage(y, dbs[0], dbs[1])
    phiz[n:] = dJdt.transpose().flatten()
    return phiz

def discretize(L, n, p, k, delta=1e-8, *args):
    print('discretizing...')
    bd = np.empty((n,n))
    Sd = np.empty(n)
    bdr = np.empty((p,n,n))
    bdl = np.empty((p,n,n))
    Sdr = np.empty((p,n))
    Sdl = np.empty((p,n))
    
    bd = breakage_discretize(L, n, k, *args)
    Sd = selection_discretize(L, n, k, bd, *args)
    
    for i in range(p):
        kr = k.copy()
        kl = k.copy()
        kr[i] += delta
        kl[i] -= delta
        bdr[i] = breakage_discretize(L, n, kr, *args)
        bdl[i] = breakage_discretize(L, n, kl, *args)
        Sdr[i] = selection_discretize(L, n, kr, bdr[i], *args)
        Sdl[i] = selection_discretize(L, n, kl, bdl[i], *args)
        
    return bd, Sd, bdr, Sdr, bdl, Sdl

def solve_jac(ode, yhat, dbs, t, n, p, N, scalar, delta):
    print('Solving ODE...')
    # initial condition J0 = 0
    if scalar:
        y0 = yhat[0]
    else:
        y0 = yhat[:,0]
        
    r = np.zeros((n, N))    
    Z0 = np.zeros(n * (p + 1))
    Z0[0:n] = y0.copy()
    
    Z, suc = evolve(ode, Z0, dbs, t, n, p, delta=1e-8)
        
    Y = Z[0:n]
    J = Z[n:]
    Jt = np.hsplit(J,N)
    
    for i in range(N):
        Jt[i] = Jt[i].reshape(p,n).transpose()
        
    return Y, Jt


if __name__ == '__main__':
    import time
    from data_import import importing
    
    k0 = np.array([1e-7,0.8,0.15])
    length, volume, number, N0, Y0, mu, sigma, t, n, N, p, Q =\
        importing(k0)
    arguments = [mu, sigma]
    tic = time.time()
    dbs = discretize(length, n, p, k0, 1e-8, *arguments)
    duration = time.time() - tic
    print('discretization takes %f seconds' % duration)
    tic = time.time()
    Y, Jac = solve_jac(breakage, number, dbs, t, n, p, N, False, delta=1e-8)
    duration = time.time() - tic
    print('solving ODE takes %f seconds' % duration)