# cython: language_level=3

import numpy as np
from ode_cy_cdef cimport breakage
from cython import boundscheck, wraparound

@boundscheck(False)
@wraparound(False)
def phi_breakage(z, dbs, Py_ssize_t n, Py_ssize_t p, double delta):
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
    cdef double[:, :] brk_view = dbs[0]
    cdef double[:] slc_view = dbs[1]
    cdef double[:, :, :] brk_view_r = dbs[2]
    cdef double[:, :] slc_view_r = dbs[3]
    cdef double[:, :, :] brk_view_l = dbs[4]
    cdef double[:, :] slc_view_l = dbs[5]
    cdef double[:] y_view = y
    cdef double[:, :] Yr_view = Yr
    cdef double[:, :] Yl_view = Yl
    cdef double[:, :] dfdy_view = dfdy
    cdef double[:, :] dfdk_view = dfdk
    
    cdef double[:] temp1 = np.empty(n)
    cdef double[:] temp2 = np.empty(n)
    
    cdef Py_ssize_t i, j
    
    for i in range(n):
        temp1 = breakage(Yr_view[i], brk_view, slc_view)
        temp2 = breakage(Yl_view[i], brk_view, slc_view)
        for j in range(n):
            dfdy_view[i, j] = (temp1[j] - temp2[j]) / (2 * delta)
            
    dfdy = dfdy.transpose()
    
    for i in range(p):
        temp1 = breakage(y_view, brk_view_r[i], slc_view_r[i])
        temp2 = breakage(y_view, brk_view_l[i], slc_view_l[i])
        for j in range(n):
            dfdk_view[i, j] = (temp1[j] - temp2[j]) / (2 * delta)
            
    dfdk = dfdk.transpose()
    
    dJdt = dfdy @ J + dfdk
    phiz[0:n] = breakage(y_view, brk_view, slc_view)
    phiz[n:] = dJdt.transpose().flatten()
    return phiz