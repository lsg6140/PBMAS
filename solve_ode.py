import numpy as np
from scipy.integrate import solve_ivp

def integ_breakage(breakage, z0, dbs, t, n, p, delta):
    def dzdt(t, z):
        return phi_breakage(breakage, z, dbs, n, p, delta)
    solution = solve_ivp(dzdt, [t[0],t[-1]], z0, method = 'Radau', t_eval = t)
    return solution.y, solution.success



def integ_breakage_onestep(breakage, z, dbs, t, n, p, delta):
    def dxdt(t, x):
        return phi_breakage(breakage, x, dbs, n, p, delta)
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



def discretize(bfunc, Sfunc, L, n, p, k, delta):
    print('discretizing')
    bd = np.empty((n,n))
    Sd = np.empty(n)
    bdr = np.empty((p,n,n))
    bdl = np.empty((p,n,n))
    Sdr = np.empty((p,n))
    Sdl = np.empty((p,n))
    
    bd = breakage_discretize(Sfunc, bfunc, L, n, k)
    Sd = selection_discretize(Sfunc, bfunc, L, n, k, bd)
    
    for i in range(p):
        kr = k.copy()
        kl = k.copy()
        kr[i] += delta
        kl[i] -= delta
        bdr[i] = breakage_discretize(Sfunc, bfunc, L, n, kr)
        bdl[i] = breakage_discretize(Sfunc, bfunc, L, n, kl)
        Sdr[i] = selection_discretize(Sfunc, bfunc, L, n, kr, bdr[i])
        Sdl[i] = selection_discretize(Sfunc, bfunc, L, n, kl, bdl[i])
        
    return bd, Sd, bdr, Sdr, bdl, Sdl