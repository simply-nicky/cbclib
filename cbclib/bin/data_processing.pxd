cdef extern from "Python.h":
    int Py_AtExit(void(*func)())

ctypedef int (*convolve_func)(double*, double*, int, unsigned long*, double*,
                              unsigned long, int, int, double, unsigned)

cdef extern from "pocket_fft.h":
    unsigned long next_fast_len_fftw(unsigned long target) nogil
    unsigned long good_size(unsigned long n) nogil

cdef extern from "fft_functions.h":
    int fft_convolve_fftw(double *out, double *inp, int ndim, unsigned long* dims, double *krn,
                          unsigned long ksize, int axis, int mode, double cval, unsigned threads) nogil

    int fft_convolve_np(double *out, double *inp, int ndim, unsigned long* dims, double *krn,
                          unsigned long ksize, int axis, int mode, double cval, unsigned threads) nogil

    int gauss_kernel1d(double *out, double sigma, unsigned order, unsigned long ksize) nogil

    int gauss_filter(double *out, double *inp, int ndim, unsigned long *dims, double *sigma,
                     unsigned *order, int mode, double cval, double truncate, unsigned threads,
                     convolve_func fft_convolve) nogil

    int gauss_grad_mag(double *out, double *inp, int ndim, unsigned long *dims, double *sigma,
                       int mode, double cval, double truncate, unsigned threads,
                       convolve_func fft_convolve) nogil

cdef extern from "median.h":
    int compare_double(void *a, void *b) nogil
    int compare_float(void *a, void *b) nogil
    int compare_long(void *a, void *b) nogil

    int median_c "median" (void *out, void *data, unsigned char *mask, int ndim, unsigned long *dims,
                 unsigned long item_size, int axis, int (*compar)(void*, void*), unsigned threads) nogil

    int median_filter_c "median_filter" (void *out, void *data, unsigned char *mask, int ndim,
                        unsigned long *dims, unsigned long item_size, int axis, unsigned long window,
                        int mode, void *cval, int (*compar)(void*, void*), unsigned threads) nogil

cdef extern from "fftw3.h":
    void fftw_init_threads() nogil
    void fftw_cleanup() nogil
    void fftw_cleanup_threads() nogil

cdef enum:
    EXTEND_CONSTANT = 0
    EXTEND_NEAREST = 1
    EXTEND_MIRROR = 2
    EXTEND_REFLECT = 3
    EXTEND_WRAP = 4