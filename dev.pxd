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
    int compare_int(void *a, void *b) nogil
    int compare_uint(void *a, void *b) nogil

    int median_c "median" (void *out, void *data, unsigned char *mask, int ndim, unsigned long *dims,
                 unsigned long item_size, int axis, int (*compar)(void*, void*), unsigned threads) nogil

    int median_filter_c "median_filter" (void *out, void *data, unsigned char *mask, int ndim,
                        unsigned long *dims, unsigned long item_size, unsigned long *fsize,
                        unsigned char *fmask, int mode, void *cval, int (*compar)(void*, void*),
                        unsigned threads) nogil

    int maximum_filter_c "maximum_filter" (void *out, void *data, unsigned char *mask, int ndim,
                         unsigned long *dims, unsigned long item_size, unsigned long *fsize,
                         unsigned char *fmask, int mode, void *cval, int (*compar)(void*, void*),
                         unsigned threads) nogil

cdef extern from "img_proc.h":
    int draw_lines_c "draw_lines" (unsigned int *out, unsigned long Y, unsigned long X,
                                   unsigned int max_val, double *lines, unsigned long lines,
                                   unsigned int dilation) nogil

    int filter_lines_c "filter_lines" (double *olines, double *data, unsigned long Y, unsigned long X, double *lines,
                       unsigned long n_lines, double x_c, double y_c, double radius) nogil

cdef enum:
    EXTEND_CONSTANT = 0
    EXTEND_NEAREST = 1
    EXTEND_MIRROR = 2
    EXTEND_REFLECT = 3
    EXTEND_WRAP = 4