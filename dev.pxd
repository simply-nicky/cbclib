#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
cimport numpy as np
from cpython.ref cimport Py_INCREF
from libc.stdlib cimport free, malloc, calloc
from cbclib.bin.image_proc cimport check_array, normalize_sequence, median_filter_c, extend_mode_to_code
from cbclib.bin.image_proc cimport compare_double, compare_float, compare_int, compare_uint, compare_ulong
from cbclib.bin.line_detector cimport ArrayWrapper

cdef extern from "lsd.h":
    int LineSegmentDetection(float **out, int *n_out, float *img, int img_x, int img_y,
                             float scale, float sigma_scale,
                             float quant, float ang_th, float log_eps,
                             float density_th, int n_bins,
                             int **reg_img, int *reg_x, int *reg_y) nogil

cdef extern from "img_proc.h":
    int draw_lines(unsigned int *out, unsigned long Y, unsigned long X, unsigned int max_val,
                   float *lines, unsigned long lines, unsigned int dilation) nogil

    int draw_line_indices(unsigned int **out, unsigned long *n_idxs, unsigned long Y, unsigned long X,
                          unsigned int max_val, float *lines, unsigned long n_lines,
                          unsigned int dilation) nogil

    int generate_rot_matrix(double *rot_mats, double *angles, unsigned long n_mats, 
                            double a0, double a1, double a2) nogil

    int filter_lines_c "filter_lines" (float *olines, float *data, unsigned long Y,
                       unsigned long X, float *ilines, unsigned long n_lines, float x_c,
                       float y_c, float *radii, float thr) nogil

cdef enum:
    EXTEND_CONSTANT = 0
    EXTEND_NEAREST = 1
    EXTEND_MIRROR = 2
    EXTEND_REFLECT = 3
    EXTEND_WRAP = 4

cdef class LSD:
    cdef public float ang_th
    cdef public float density_th
    cdef public float log_eps
    cdef public float scale
    cdef public float sigma_scale
    cdef public float quant
    cdef public float x_c
    cdef public float y_c