import numpy as np
from cython import boundscheck, wraparound

@boundscheck(False)
@wraparound(False)
cdef double[:] breakage(double[:] number, double[:, :] brk_mat, double[:] slc_vec):
    cdef Py_ssize_t n = len(number)
    
    # Memoryview
    cdef double[:] dndt = np.zeros(n)
    
    cdef Py_ssize_t i, j
    cdef double sum
    
    sum = 0
    for j in range(n):
        sum += brk_mat[0, j] * slc_vec[j] * number[j]
    dndt[0] = sum
    
    for i in range(1, n):
        sum = 0
        for j in range(i, n):
            sum += brk_mat[i, j] * slc_vec[j] * number[j]
        dndt[i] = sum - slc_vec[i] * number[i]

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