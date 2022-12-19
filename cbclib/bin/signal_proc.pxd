cimport numpy as np
from libc.stdlib cimport free, malloc
from .image_proc cimport check_array, normalize_sequence
from .line_detector cimport ArrayWrapper

ctypedef double (*kernel)(double, double)

cdef extern from "sgn_proc.h":
    double rbf(double dist, double sigma) nogil

    int predict_kerreg(double *y, double *w, double *x, unsigned long npts, unsigned long ndim, double *y_hat,
                       double *x_hat, unsigned long nhat, kernel krn, double sigma, double cutoff,
                       unsigned threads) nogil

    int predict_grid(double *y, double *w, double *x, unsigned long npts, unsigned long ndim, double *y_hat,
                     unsigned long *roi, double *step, kernel krn, double sigma, double cutoff,
                     unsigned threads) nogil

    int unique_indices_c "unique_indices" (unsigned **funiq, unsigned **fidxs, unsigned long *fpts,
                      unsigned **iidxs, unsigned long *ipts, unsigned *frames, unsigned *indices,
                      unsigned long npts) nogil

    int update_sf_c "update_sf" (float *sf, float *dsf, float *bp, float *sgn, unsigned *xidx, float *xmap,
                    float *xtal, unsigned long *ddims, unsigned *hkl_idxs, unsigned long hkl_size,
                    unsigned *iidxs, unsigned long isize, unsigned threads) nogil

    float scale_crit(float *sf, float *bp, float *sgn, unsigned *xidx, float *xmap, float *xtal, unsigned long *ddims,
                     unsigned *iidxs, unsigned long isize, unsigned threads) nogil

    int xtal_interp(float *xtal_bi, unsigned *xidx, float *xmap, float *xtal, unsigned long *ddims,
                    unsigned long isize, unsigned threads) nogil
