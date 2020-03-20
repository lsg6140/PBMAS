import numpy as np

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