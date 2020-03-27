import numpy as np
from joblib import Parallel, delayed

def phi_breakage(breakage, z, dbs, n, p, delta):
    # dbs: discretized breakage and selection functions
    z = z.astype(np.float)
    y = z[0:n]
    J = z[n:].reshape((p, n)).transpose()
    phiz = np.empty(n * (p + 1))
    
    Y = np.tile(y, [n, 1])
    Yr = Y + np.eye(n) * delta
    Yl = Y - np.eye(n) * delta
    
    def in_for_loop1(i):
        return (breakage(Yr[i], dbs[0], dbs[1]) - breakage(Yl[i], dbs[0], dbs[1])) / (2 * delta)
    
    
    def in_for_loop2(i):
        return (breakage(y, dbs[2][i], dbs[3][i]) - breakage(y, dbs[4][i], dbs[5][i])) / (2 * delta)
        
    r1 = Parallel(n_jobs=-1)(delayed(in_for_loop1)(i) for i in range(n))
    
    dfdy = np.stack(r1).T
    
    r2 = Parallel(n_jobs=-1)(delayed(in_for_loop2)(i) for i in range(p))
    
    dfdk = np.stack(r2).T
   
    dJdt = dfdy @ J + dfdk
    phiz[0:n] = breakage(y, dbs[0], dbs[1])
    phiz[n:] = dJdt.transpose().flatten()
    return phiz