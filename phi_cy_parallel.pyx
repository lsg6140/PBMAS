# cython: language_level=3
# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp

import numpy as np

from cython.parallel import prange 
from cython cimport boundscheck, wraparound
from ode_nogil cimport breakage

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
    
    # Memorview
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
    cdef double[:] dndt_view = phiz[:n]
    
    cdef double[:, :] temp1 = np.empty((n, n))
    cdef double[:, :] temp2 = np.empty((n, n))
    
    cdef Py_ssize_t i, j
    
    for i in prange(n, nogil=True):
        breakage(temp1[i], Yr_view[i], brk_view, slc_view)
        breakage(temp2[i], Yl_view[i], brk_view, slc_view)
        for j in range(n):
            dfdy_view[i, j] = (temp1[i, j] - temp2[i, j]) / (2 * delta)
            
    dfdy = dfdy.transpose()
    
    for i in prange(p, nogil=True):
        breakage(temp1[i], y_view, brk_view_r[i], slc_view_r[i])
        breakage(temp2[i], y_view, brk_view_l[i], slc_view_l[i])
        for j in range(n):
            dfdk_view[i, j] = (temp1[i, j] - temp2[i, j]) / (2 * delta)
            
    dfdk = dfdk.transpose()
    
    dJdt = dfdy @ J + dfdk
    breakage(dndt_view, y_view, brk_view, slc_view)
    phiz[n:] = dJdt.transpose().flatten()
    return phiz