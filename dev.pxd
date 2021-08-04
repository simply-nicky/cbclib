#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
cdef extern from "lsd.h":
    int LineSegmentDetection(double **out, int *n_out, double *img, int img_x, int img_y,
                             double scale, double sigma_scale,
                             double quant, double ang_th, double log_eps,
                             double density_th, int n_bins,
                             int **reg_img, int *reg_x, int *reg_y) nogil

cdef extern from "median.h":
    int compare_double(void *a, void *b) nogil
    int compare_float(void *a, void *b) nogil
    int compare_long(void *a, void *b) nogil

    int median_c "median" (void *out, void *data, unsigned char *mask, int ndim, unsigned long *dims,
                 unsigned long item_size, int axis, int (*compar)(void*, void*), unsigned threads) nogil

    # int median_filter_c "median_filter" (void *out, void *data, unsigned char *mask, int ndim,
    #                     unsigned long *dims, unsigned long item_size, int axis, unsigned long window,
    #                     int mode, void *cval, int (*compar)(void*, void*), unsigned threads) nogil

    int median_filter_c "median_filter" (void *out, void *data, int ndim, unsigned long *dims,
                        unsigned long item_size, unsigned long *fsize, int mode, void *cval,
                        int (*compar)(void*, void*), unsigned threads) nogil

cdef enum:
    EXTEND_CONSTANT = 0
    EXTEND_NEAREST = 1
    EXTEND_MIRROR = 2
    EXTEND_REFLECT = 3
    EXTEND_WRAP = 4