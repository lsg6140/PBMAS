# cython: language_level=3
cdef double[:] breakage(double[:] number, double[:, :] brk_mat, double[:] slc_vec)