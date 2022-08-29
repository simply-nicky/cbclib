#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
cimport numpy as np
from cpython.ref cimport Py_INCREF
from libc.stdlib cimport free, malloc, calloc, realloc
from libc.string cimport memset
from cbclib.bin.image_proc cimport check_array, normalize_sequence, median_filter_c, extend_mode_to_code
from cbclib.bin.image_proc cimport compare_double, compare_float, compare_int, compare_uint, compare_ulong
from cbclib.bin.line_detector cimport ArrayWrapper

cdef extern from "array.h":
    int compare_double(void *a, void *b) nogil
    int compare_float(void *a, void *b) nogil
    int compare_int(void *a, void *b) nogil
    int compare_uint(void *a, void *b) nogil
    int compare_ulong(void *a, void *b) nogil

ctypedef double (*kernel)(double, double)

cdef extern from "smoothers.h":
    double rbf(double dist, double sigma) nogil

    int predict_kerreg(double *y, double *x, unsigned long npts, unsigned long ndim, double *y_hat,
                       double *x_hat, unsigned long nhat, kernel krn, double sigma, double cutoff,
                       double epsilon, unsigned threads) nogil