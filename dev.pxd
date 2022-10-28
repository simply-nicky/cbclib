#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
cimport numpy as np
from cpython.ref cimport Py_INCREF
from libc.stdlib cimport free, malloc, calloc, realloc
from libc.string cimport memset
from cbclib.bin.image_proc cimport check_array, normalize_sequence, median_filter_c, mode_to_code
from cbclib.bin.cbc_indexing cimport side_to_code
from cbclib.bin.line_detector cimport ArrayWrapper

cdef enum:
    SEARCH_LEFT = 0
    SEARCH_RIGHT = 1

cdef extern from "array.h":
    int compare_double(void *a, void *b) nogil
    int compare_float(void *a, void *b) nogil
    int compare_int(void *a, void *b) nogil
    int compare_uint(void *a, void *b) nogil
    int compare_ulong(void *a, void *b) nogil

    unsigned long searchsorted_c "searchsorted" (void *key, void *base, unsigned long npts,
                                 unsigned long size, int side, int (*compar)(void*, void*)) nogil

ctypedef double (*kernel)(double, double)

cdef extern from "sgn_proc.h":
    double rbf(double dist, double sigma) nogil

    int predict_kerreg(double *y, double *w, double *x, unsigned long npts, unsigned long ndim, double *y_hat,
                       double *x_hat, unsigned long nhat, kernel krn, double sigma, double cutoff,
                       unsigned threads) nogil

    int predict_grid(double *y, double *w, double *x, unsigned long npts, unsigned long ndim, double *y_hat,
                     unsigned long *dims, double *step, kernel krn, double sigma, double cutoff,
                     unsigned threads) nogil

    int index_table_c "index_table" (unsigned **funiq, unsigned **fidxs, unsigned long *fpts,
                      unsigned **iidxs, unsigned long *ipts, unsigned *frames, unsigned *indices,
                      unsigned long npts) nogil

    int update_sf_c "update_sf" (float *sf, float *sgn, unsigned *frames, float *kins, float *dmap,
                    unsigned long *ddims, unsigned *hkl_idxs, unsigned long hkl_size, unsigned *iidxs,
                    unsigned long isize, unsigned threads) nogil

    float scale_crit(float *sf, float *sgn, unsigned *frames, float *kins, float *dmap, unsigned long *ddims,
                     unsigned *iidxs, unsigned long isize, unsigned threads) nogil

ctypedef int (*line_profile)(int, float, float)

cdef extern from "img_proc.h":
    int linear_profile(int max_val, float err, float wd) nogil
    int tophat_profile(int max_val, float err, float wd) nogil
    int quad_profile(int max_val, float err, float wd) nogil    
    
    int draw_line_c "draw_line" (unsigned int *out, unsigned long Y, unsigned long X, unsigned int max_val,
                    float *lines, unsigned long *ldims, float dilation, line_profile profile) nogil

    int draw_line_index_c "draw_line_index" (unsigned int **out, unsigned long *n_idxs, unsigned long Y,
                          unsigned long X, unsigned int max_val, float *lines, unsigned long *ldims,
                          float dilation, line_profile profile) nogil

cdef extern from "lsd.h":
    int LineSegmentDetection(float **out, int *n_out, float *img, int img_x, int img_y,
                             float scale, float sigma_scale,
                             float quant, float ang_th, float log_eps,
                             float density_th, int n_bins,
                             int **reg_img, int *reg_x, int *reg_y) nogil

cdef extern from "img_proc.h":
    int filter_line(float *olines, unsigned char *proc, float *data, unsigned long Y, unsigned long X,
                     float *ilines, unsigned long *ldims, float threshold, float dilation) nogil

    int group_line(float *olines, unsigned char *proc, float *data, unsigned long Y, unsigned long X,
                    float *ilines, unsigned long *ldims, float cutoff, float threshold, float dilation) nogil

cdef extern from "median.h":
    int median_c "median" (void *out, void *data, unsigned char *mask, int ndim, unsigned long *dims,
                 unsigned long item_size, int axis, int (*compar)(void*, void*), unsigned threads) nogil

    int median_filter_c "median_filter" (void *out, void *data, unsigned char *mask, unsigned char *imask,
                        int ndim, unsigned long *dims, unsigned long item_size, unsigned long *fsize,
                        unsigned char *fmask, int mode, void *cval, int (*compar)(void*, void*),
                        unsigned threads) nogil

    int maximum_filter_c "maximum_filter" (void *out, void *data, unsigned char *mask, unsigned char *imask,
                         int ndim, unsigned long *dims, unsigned long item_size, unsigned long *fsize,
                         unsigned char *fmask, int mode, void *cval, int (*compar)(void*, void*),
                         unsigned threads) nogil

cdef class LSD:
    cdef public float ang_th
    cdef public float density_th
    cdef public float log_eps
    cdef public float scale
    cdef public float sigma_scale
    cdef public float quant
    cdef public float x_c
    cdef public float y_c
