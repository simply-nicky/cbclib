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
    int compute_euler_angles(float *angles, float *rot_mats, unsigned long n_mats) nogil
    int compute_euler_matrix(float *rot_mats, float *angles, unsigned long n_mats) nogil
    int compute_tilt_angles(float *angles, float *rot_mats, unsigned long n_mats) nogil
    int compute_tilt_matrix(float *rot_mats, float *angles, unsigned long n_mats) nogil
    int compute_rotations(float *rot_mats, float *as, float *bs, unsigned long n_mats) nogil

    int det2k(float *karr, float *x, float *y, unsigned *idxs, unsigned long ksize, float *src,
              unsigned threads) nogil
    int k2det(float *x, float *y, float *karr, unsigned *idxs, unsigned long ksize, float *src,
              unsigned threads) nogil
    int k2smp(float *pts, float *karr, unsigned *idxs, unsigned long ksize, float *z, float *src,
              unsigned threads) nogil
    int rotate_vec(float *out, float *vecs, unsigned *idxs, unsigned long vsize, float *rmats,
                   unsigned threads) nogil