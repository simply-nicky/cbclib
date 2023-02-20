#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
cimport numpy as np
cimport openmp
from cpython.ref cimport Py_INCREF
from libc.stdlib cimport free, malloc, calloc, realloc
from libc.string cimport memset
from cbclib.bin.image_proc cimport check_array, normalize_sequence, median_filter_c, mode_to_code
from cbclib.bin.line_detector cimport ArrayWrapper

cdef enum:
    SEARCH_LEFT = 0
    SEARCH_RIGHT = 1

cdef inline int side_to_code(str side) except -1:
    if side == 'left':
        return SEARCH_LEFT
    elif side == 'right':
        return SEARCH_RIGHT
    else:
        raise RuntimeError(f'Invalid side keyword: {side}')

cdef extern from "array.h":
    int compare_double(void *a, void *b) nogil
    int compare_float(void *a, void *b) nogil
    int compare_int(void *a, void *b) nogil
    int compare_uint(void *a, void *b) nogil
    int compare_ulong(void *a, void *b) nogil

    unsigned long searchsorted(void *key, void *base, unsigned long npts, unsigned long size, int side,
                               int (*compar)(void*, void*)) nogil

ctypedef float (*kernel)(float, float)
ctypedef float (*loss_func)(float)

cdef extern from "sgn_proc.h":
    int interp_bi(float *out, float *data, int ndim, unsigned long *dims, float **grid, float *crds,
                  unsigned long ncrd, unsigned threads) nogil

    float rbf(float dist, float sigma) nogil

    int predict_kerreg(float *y, float *w, float *x, unsigned long npts, unsigned long ndim, float *y_hat,
                       float *x_hat, unsigned long nhat, kernel krn, float sigma, unsigned threads) nogil

    int predict_grid(float **y_hat, unsigned long *roi, float *y, float *w, float *x, unsigned long npts,
                     unsigned long ndim, float **grid, unsigned long *gdims, kernel krn, float sigma,
                     unsigned threads) nogil

    int unique_indices_c "unique_indices" (unsigned **funiq, unsigned **fidxs, unsigned long *fpts,
                      unsigned **iidxs, unsigned long *ipts, unsigned *frames, unsigned *indices,
                      unsigned long npts) nogil

    float l2_loss(float x) nogil
    float l2_grad(float x) nogil
    float l1_loss(float x) nogil
    float l1_grad(float x) nogil
    float huber_loss(float x) nogil
    float huber_grad(float x) nogil

    float poisson_likelihood(float *grad, float *x, unsigned long xsize, float *rp, unsigned *I0, float *bgd,
                             float *xtal_bi, unsigned *hkl_idxs, unsigned *iidxs, unsigned long isize,
                             unsigned threads) nogil

    float least_squares(float *grad, float *x, unsigned long xsize, float *rp, unsigned *I0, float *bgd,
                        float *xtal_bi, unsigned *hkl_idxs, unsigned *iidxs, unsigned long isize,
                        loss_func func, loss_func grad, unsigned threads) nogil

ctypedef float (*line_profile)(float, float)

cdef extern from "img_proc.h":
    float linear_profile(float err, float wd) nogil
    float tophat_profile(float err, float wd) nogil
    float quad_profile(float err, float wd) nogil
    float gauss_profile(float err, float wd) nogil
    
    int draw_line_int(unsigned *out, unsigned long *dims, unsigned max_val, float *lines,
                      unsigned long *ldims, float dilation, line_profile profile) nogil

    int draw_line_float(float *out, unsigned long *dims, float *lines, unsigned long *ldims,
                        float dilation, line_profile profile) nogil

    int draw_line_index(unsigned **idx, unsigned **x, unsigned **y, float **val, unsigned long *n_idxs,
                        unsigned long *dims, float *lines, unsigned long *ldims, float dilation,
                        line_profile profile) nogil

    int filter_line(float *olines, unsigned char *proc, float *data, unsigned long *dims, float *ilines,
                    unsigned long *ldims, float threshold, float dilation, line_profile profile) nogil

    int group_line(float *olines, unsigned char *proc, float *data, unsigned long *dims, float *ilines,
                   unsigned long *ldims, float cutoff, float threshold, float dilation, line_profile profile) nogil

    int normalise_line(float *out, float *data, unsigned long *dims, float *lines, unsigned long *ldims,
                       float *dilations, line_profile profile) nogil

    int refine_line(float *olines, float *data, unsigned long *dims, float *ilines, unsigned long *ldims,
                    float dilation, line_profile profile) nogil

    double cross_entropy(unsigned *ij, float *p, unsigned *fidxs, unsigned long *dims, float **lines,
                         unsigned long *ldims, unsigned long lsize, float dilation, float epsilon,
                         line_profile profile, unsigned threads) nogil

cdef extern from "lsd.h":
    int LineSegmentDetection(float **out, int *n_out, float *img, unsigned long *dims, float scale,
                             float sigma_scale, float quant, float ang_th, float log_eps,
                             float density_th, int n_bins, int **reg_img, int *reg_x, int *reg_y) nogil

cdef extern from "median.h":
    int median_c "median" (void *out, void *data, unsigned char *mask, int ndim, unsigned long *dims,
                 unsigned long item_size, int axis, int (*compar)(void*, void*), unsigned threads) nogil

    int median_filter_c "median_filter" (void *out, void *data, unsigned char *mask, unsigned char *imask,
                        int ndim, unsigned long *dims, unsigned long item_size, unsigned long *fsize,
                        unsigned char *fmask, int mode, void *cval, int (*compar)(void*, void*),
                        unsigned threads) nogil

    int maximum_filter_c "maximum_filter" (void *out, void *data, unsigned char *mask, unsigned char *imask,
                         int ndim, unsigned long *dims, unsigned long item_size, unsigned long *fsize,
                         unsigned char *fmask, int mode, void *cval, int (*compar)(void *, void *),
                         unsigned threads) nogil
