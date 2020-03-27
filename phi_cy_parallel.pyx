# cython: language_level=3

import numpy as np

from cython.parallel import prange 

def phi_breakage(breakage, z, dbs, Py_ssize_t n, Py_ssize_t p, double delta):
    # dbs: discretized breakage and selection functions
    z = z.astype(np.float)
    y = z[0:n]
    J = z[n:].reshape((p, n)).transpose()
    phiz = np.empty(n * (p + 1))
    dfdy = np.empty((n, n))
    dfdk = np.empty((p, n))
    
    Y = np.tile(y, [n, 1])
    Yr = Y + np.eye(n) * delta
    Yl = Y - np.eye(n) * delta
    
    # Memoryview
    cdef double[:, :] brk_mat = dbs[0]
    cdef double[:] slc_vec = dbs[1]
    cdef double[:, :, :] brk_mat_r = dbs[2]
    cdef double[:, :] slc_vec_r = dbs[3]
    cdef double[:, :, :] brk_mat_l = dbs[4]
    cdef double[:, :] slc_vec_l = dbs[5]
    cdef double[:] yv = y
    cdef double[:, :] Yrv = Yr
    cdef double[:, :] Ylv = Yl
    cdef double[:, :] dfdyv = dfdy
    cdef double[:, :] dfdkv = dfdk
    
    cdef double[:] temp1
    cdef double[:] temp2
    
    cdef Py_ssize_t i, j
    
    for i in prange(n, nogil=True):
        temp1 = breakage(Yr[i], brk_mat, slc_vec)
        temp2 = breakage(Yl[i], brk_mat, slc_vec)
        for j in range(n):
            dfdyv[i, j] = (temp1[j] - temp2[j]) / (2 * delta)
            
    dfdy = dfdy.transpose()
    
    for i in prange(p, nogil=True):
        temp1 = breakage(yv, brk_mat_r[i], slc_vec_r[i])
        temp2 = breakage(yv, brk_mat_l[i], slc_vec_l[i])
        for j in range(n):
            dfdkv[i, j] = (temp1[j] - temp2[j]) / (2 * delta)
            
    dfdk = dfdk.transpose()
    
    dJdt = dfdy @ J + dfdk
    phiz[0:n] = breakage(y, brk_mat, slc_vec)
    phiz[n:] = dJdt.transpose().flatten()
    return phiz