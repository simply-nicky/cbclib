cimport numpy as np
from libc.math cimport exp
from libc.stdlib cimport free, malloc
from libc.string cimport memcpy

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
