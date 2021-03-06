# cython: language_level=3

import numpy as np
from cython import boundscheck, wraparound

@boundscheck(False)
@wraparound(False)
def breakage(number, brk_mat, slc_vec):
    cdef Py_ssize_t n = len(number)
    R1 = np.zeros(n)
    R2 = np.zeros(n)
    
    # Memoryview
    cdef double[:] R1_view = R1
    cdef double[:] R2_view = R2
    cdef double[:] n_view = number
    cdef double[:, :] brk_view = brk_mat
    cdef double[:] slc_view = slc_vec
    
    cdef Py_ssize_t i, j
    cdef double sum
    
    # Mechanism 1 (i=1~n, j=i~n) !!! with index 1~n
    for i in range(n):
        sum = 0
        for j in range(i, n):
            sum += brk_view[i, j] * slc_view[j] * n_view[j]
        R1_view[i] = sum
        
    # Mechanism 2 (i=2~n)
    for i in range(1, n):
        R2_view[i] = slc_view[i] * n_view[i]

    return R1 - R2



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