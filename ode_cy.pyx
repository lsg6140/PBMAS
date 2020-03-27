# cython: language_level=3

import numpy as np

cpdef double[:] breakage(double[:] number, double[:, :] brk_mat, double[:] slc_vec):
    cdef Py_ssize_t n = len(number)
    cdef double[:] dndt = np.zeros(n).astype(np.double)
    
    cdef Py_ssize_t i, j
    cdef double sum
    
    # Mechanism 1 (i=1~n, j=i~n) !!! with index 1~n
    for i in range(n):
        sum = 0
        for j in range(i, n):
            sum += brk_mat[i, j] * slc_vec[j] * number[j]
        dndt[i] = sum
        
    # Mechanism 2 (i=2~n)
    for i in range(1, n):
        dndt[i] -= slc_vec[i] * number[i]

    return dndt



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