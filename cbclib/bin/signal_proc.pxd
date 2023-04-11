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

    int unique_idxs(unsigned **unique, unsigned **iidxs, unsigned long *isize, unsigned *indices,
                    unsigned *inverse, unsigned long npts) nogil

    float l2_loss(float x) nogil
    float l2_grad(float x) nogil
    float l1_loss(float x) nogil
    float l1_grad(float x) nogil
    float huber_loss(float x) nogil
    float huber_grad(float x) nogil

    int poisson_likelihood(double *out, double *grad, float *x, unsigned *ij, unsigned long *dims, unsigned *I0,
                           float *bgd, float *xtal_bi, float *rp, unsigned *fidxs, unsigned long fsize,
                           unsigned *idxs, unsigned long isize, unsigned *hkl_idxs, unsigned long hkl_size,
                           unsigned *odixs, unsigned long osize, unsigned threads) nogil

    int least_squares(double *out, double *grad, float *x, unsigned *ij, unsigned long *dims, unsigned *I0,
                      float *bgd, float *xtal_bi, float *rp, unsigned *fidxs, unsigned long fsize,
                      unsigned *idxs, unsigned long isize, unsigned *hkl_idxs, unsigned long hkl_size,
                      unsigned *oidxs, unsigned long osize, loss_func func, loss_func grad, unsigned threads) nogil

    int unmerge_sgn(float *I_hat, float *x, unsigned *ij, unsigned long *dims, unsigned *I0, float *bgd,
                    float *xtal_bi, float *rp, unsigned *fidxs, unsigned long fsize, unsigned *idxs,
                    unsigned long isize, unsigned *hkl_idxs, unsigned long hkl_size, unsigned threads) nogil
