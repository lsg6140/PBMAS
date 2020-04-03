# cython: language_level=3
# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp

import numpy as np
from cython.parallel cimport prange
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
    
    for i in prange(n, nogil=True):
        sum = 0
        for j in range(i, n):
            sum = sum + brk_view[i, j] * slc_view[j] * n_view[j]
        R1_view[i] = sum
        R2_view[i] = slc_view[i] * n_view[i]
        
    R2_view[0] = 0.0

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