# cython: language_level=3

cdef void breakage(double[:] dndt, double[:] number, double[:, :] brk_mat, double[:] slc_vec) nogil
