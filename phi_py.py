import numpy as np
from ode import breakage

def phi_breakage(z, dbs, n, p, delta):
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