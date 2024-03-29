cimport numpy as np
cimport openmp
from libc.stdlib cimport free, calloc, malloc
from libc.math cimport exp, sqrt, cos, sin, tan, acos, atan2, fabs, log
from libc.float cimport DBL_EPSILON

cdef enum:
    SEARCH_LEFT = 0
    SEARCH_RIGHT = 1

cdef extern from "array.h":
    int compare_double(void *a, void *b) nogil
    int compare_float(void *a, void *b) nogil
    int compare_int(void *a, void *b) nogil
    int compare_uint(void *a, void *b) nogil
    int compare_ulong(void *a, void *b) nogil

    unsigned long searchsorted(void *key, void *base, unsigned long npts, unsigned long size,
                               int side, int (*compar)(void*, void*)) nogil

cdef extern from "geometry.h":
    int compute_euler_angles(double *angles, double *rot_mats, unsigned long n_mats) nogil
    int compute_euler_matrix(double *rot_mats, double *angles, unsigned long n_mats) nogil
    int compute_tilt_angles(double *angles, double *rot_mats, unsigned long n_mats) nogil
    int compute_tilt_matrix(double *rot_mats, double *angles, unsigned long n_mats) nogil
    int compute_rotations(double *rot_mats, double *as, double *bs, unsigned long n_mats) nogil

    int det2k(double *karr, double *x, double *y, unsigned *idxs, unsigned long ksize, double *src,
              unsigned threads) nogil
    int k2det(double *x, double *y, double *karr, unsigned *idxs, unsigned long ksize, double *src,
              unsigned threads) nogil
    int k2smp(double *pts, double *karr, unsigned *idxs, unsigned long ksize, double *z, double *src,
              unsigned threads) nogil
    int rotate_vec(double *out, double *vecs, unsigned *idxs, unsigned long vsize, double *rmats,
                   unsigned threads) nogil
