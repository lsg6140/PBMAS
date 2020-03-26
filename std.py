import numpy as np
from scipy.integrate import quad,dblquad
from numpy import linalg as LA
from scipy.integrate import solve_ivp

def breakage(N, bmat, Svec):
    # N is vector of particle numbers and moments
    # b is discretized breakage function
    # S is discretized selection rate
    n = len(N)
    R1 = np.zeros(n)
    
    # Mechanism 1 (i=1~n, j=i~n) !!! with index 1~n
    for i in range(n):
        R1[i] = np.sum(bmat[i, i:] * Svec[i:] * N[i:])
        
    # Mechanism 2 (i=2~n)
    R2 = Svec[1:] * N[1:]
    R2 = np.insert(R2, 0, 0.0)
        
    dNdt = R1 - R2

    return dNdt



def breakage_moment(Y, bmat, Svec, L):
    n = len(Y) - 4
    N = Y[0:n]

    dNdt = breakage(N, bmat, Svec)

    m0 = np.sum(dNdt)
    m1 = np.sum(L @ dNdt)
    m2 = np.sum(np.power(L, 2) @ dNdt)
    m3 = np.sum(np.power(L, 3) @ dNdt)
    
    dydt = np.append(dNdt,[m0,m1,m2,m3])
    
    return dydt

def breakage_discretize(Sfunc, bfunc, L, n, k):
    L = np.insert(L, 0, 0)
    bd = np.zeros((n, n))
    
    def num_func(x,y):
        return x**3 * Sfunc(y, k) * bfunc(x, y, k)
    
    def den_func(x):
        return x**3 * Sfunc(x, k)
    
    for i in range(n):
        den, err = quad(den_func, L[i], L[i+1])
        assert den != 0, 'breakage_discretize: division by zero'
        for j in range(i):
            num, err = dblquad(num_func, L[i], L[i+1],
                               lambda x: L[j], lambda x: L[j+1])
            Li = (L[i]+L[i+1])/2
            Lj = (L[j]+L[j+1])/2
            bd[j, i] = (Li / Lj)**3 * num / den
        num, err = dblquad(num_func, L[i], L[i+1],
                           lambda x: L[i], lambda x: x)
        bd[i, i] = num / den
        
    return bd 



def particle_number(bfunc, y, k): 
    Nb, err = quad(lambda x:bfunc(x, y, k), 0, y)
    return Nb



def selection_discretize(Sfunc, bfunc, L, n, k, bd):
    Sd = np.empty(n)
    L = np.insert(L, 0, 0)
    
    def integrand(y):
        int = (particle_number(bfunc, y, k) - 1) * Sfunc(y, k)
        return int
    
    for i in range(1, n):
        integ, err = quad(integrand, L[i], L[i+1])
        num = integ / (L[i+1] - L[i])
        sum = np.sum(bd[:i+1, i])
        den = sum - 1
        assert den != 0, 'selection_discretize: division by zero'
        Sd[i] = num / den
        
    Sd[0] = 0.0
    return Sd


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



def discretize(Sfunc, bfunc, L, n, p, k, delta=1e-8):
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

def breakagefunc(x, y, k):
    return x**2 * y * k[1]

def selectionfunc(y, k):
    return k[0] * y**3


if __name__ == '__main__':   
    n = 10
    k = [1.0, 2.0]
    p = len(k)
    L = np.linspace(1.0, 20, n)
    res = discretize(selectionfunc, breakagefunc, L, n, p, k, delta=1e-8)