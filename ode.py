import numpy as np

def breakage(number, brk_mat, slc_vec):
    n = len(number)
    R1 = np.zeros(n)
    
    # Mechanism 1 (i=1~n, j=i~n) !!! with index 1~n
    for i in range(n):
        R1[i] = np.sum(brk_mat[i, i:] * slc_vec[i:] * number[i:])
        
    # Mechanism 2 (i=2~n)
    R2 = slc_vec[1:] * number[1:]
    R2 = np.insert(R2, 0, 0.0)
        
    dNdt = R1 - R2

    return dNdt



def breakage_moment(Y, brk_mat, slc_vec, L):
    n = len(Y) - 4
    number = Y[0:n]

    dNdt = breakage(number, brk_mat, slc_vec)

    m0 = np.sum(dNdt)
    m1 = np.sum(L @ dNdt)
    m2 = np.sum(np.power(L, 2) @ dNdt)
    m3 = np.sum(np.power(L, 3) @ dNdt)
    
    dydt = np.append(dNdt,[m0,m1,m2,m3])
    
    return dydt