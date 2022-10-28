#ifndef SGN_PROC_H
#define SGN_PROC_H
#include "include.h"

typedef double (*kernel)(double dist, double sigma);

double rbf(double dist, double sigma);

/*-------------------------------------------------------------------------------*/
/** Predict a smoothed weighted curve with Nadaraya-Watson kernel regression.

    @param y            Buffer of input y values.
    @param w            Buffer of input weight values
    @param x            Buffer of input x values.
    @param npts         Number of (y, x) points.
    @param ndim         Number of dimensions of x.
    @param yhat         Output buffer, where the predicted curve is written.
    @param xhat         Points, where the prediction is calculated.
    @param nhat         Number of xhat points.
    @param krn          Kernel function krn(dist, sigma) of distance dist and kernel
                        bandwidth sigma.
    @param sigma        Kernel bandwidth.
    @param cutoff       Prediction is calculated based on points from [x - cutoff,
                        x + cutoff] range.
    @param threads      Number of threads used during the calculation.

    @return             Returns 0 if it has finished normally, 1 otherwise.
 */
int predict_kerreg(double *y, double *w, double *x, size_t npts, size_t ndim, double *yhat, double *xhat, size_t nhat,
                   kernel krn, double sigma, double cutoff, unsigned threads);

/*--------------------------------------------------------------------------------------*/
/** Predict a smoothed weighted curve over a grid with Nadaraya-Watson kernel regression.

    @param y            Buffer of input y values.
    @param w            Buffer of input weight values
    @param x            Buffer of input x values.
    @param npts         Number of (y, x) points.
    @param ndim         Number of dimensions of x.
    @param yhat         Output buffer, where the predicted curve is written.
    @param dims         Shape of a grid of predicted points. Number of dimensions is ndim.
    @param step         Step size of the grid. Number of dimensions is ndim.
    @param krn          Kernel function krn(dist, sigma) of distance dist and kernel
                        bandwidth sigma.
    @param sigma        Kernel bandwidth.
    @param cutoff       Prediction is calculated based on points from [x - cutoff,
                        x + cutoff] range.
    @param threads      Number of threads used during the calculation.

    @return             Returns 0 if it has finished normally, 1 otherwise.
 */
int predict_grid(double *y, double *w, double *x, size_t npts, size_t ndim, double *y_hat, const size_t *dims, double *step,
                 kernel krn, double sigma, double cutoff, unsigned threads);

int unique_indices(unsigned **funiq, unsigned **fidxs, size_t *fpts, unsigned **iidxs, size_t *ipts, unsigned *frames,
                unsigned *indices, size_t npts);

int update_sf(float *sf, float *sgn, unsigned *xidx, float *xmap, float *xtal, const size_t *ddims,
              unsigned *hkl_idxs, size_t hkl_size, unsigned *iidxs, size_t isize, unsigned threads);

float scale_crit(float *sf, float *sgn, unsigned *xidx, float *xmap, float *xtal, const size_t *ddims, unsigned *iidxs,
                 size_t isize, unsigned threads);

#endif