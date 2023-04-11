#ifndef SGN_PROC_H
#define SGN_PROC_H
#include "include.h"

/*----------------------------------------------------------------------------*/
/*------------------------- Bilinear interpolation ---------------------------*/
/*----------------------------------------------------------------------------*/
int interp_bi(float *out, float *data, int ndim, const size_t *dims, float **grid, float *crds, size_t ncrd,
              unsigned threads);

/*----------------------------------------------------------------------------*/
/*---------------------------- Kernel regression -----------------------------*/
/*----------------------------------------------------------------------------*/

typedef float (*kernel)(float dist, float sigma);

static inline float rbf(float dist, float sigma)
{
    return exp(-0.5 * SQ(dist) / SQ(sigma)) * M_1_SQRT2PI;
}

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
    @param threads      Number of threads used during the calculation.

    @return             Returns 0 if it has finished normally, 1 otherwise.
 */
int predict_kerreg(float *y, float *w, float *x, size_t npts, size_t ndim, float *yhat, float *xhat, size_t nhat,
                   kernel krn, float sigma, unsigned threads);

/*--------------------------------------------------------------------------------------*/
/** Predict a smoothed weighted curve over a grid with Nadaraya-Watson kernel regression.

    @param yhat         Output buffer, where the predicted curve is written.
    @param y            Buffer of input y values.
    @param w            Buffer of input weight values
    @param x            Buffer of input x values.
    @param npts         Number of (y, x) points.
    @param ndim         Number of dimensions of x.
    @param gdims        Shape of a grid of predicted points. Number of dimensions is ndim.
    @param krn          Kernel function krn(dist, sigma) of distance dist and kernel
                        bandwidth sigma.
    @param sigma        Kernel bandwidth.
    @param threads      Number of threads used during the calculation.

    @return             Returns 0 if it has finished normally, 1 otherwise.
 */
int predict_grid(float **y_hat, size_t *roi, float *y, float *w, float *x, size_t npts, size_t ndim, float **grid,
                 const size_t *gdims, kernel krn, float sigma, unsigned threads);

int unique_idxs(unsigned **unique, unsigned **iidxs, size_t *isize, unsigned *indices, unsigned *inverse, size_t npts);

/*----------------------------------------------------------------------------*/
/*---------------------- Intensity scaling criterions ------------------------*/
/*----------------------------------------------------------------------------*/
static inline float l2_loss(float x) {return SQ(x);}
static inline float l2_grad(float x) {return 2.0f * x;}
static inline float l1_loss(float x) {return fabsf(x);}
static inline float l1_grad(float x) {return (x > 0.0f) ? 1.0f : -1.0f;}
static inline float huber_loss(float x) {float xx = fabsf(x); return (xx < 1.345f) ? SQ(xx) : 2.69f * (xx - 0.7625f);}
static inline float huber_grad(float x) {float xx = fabsf(x); return (xx < 1.345f) ? 2.0f * x : 2.69f * x / xx;}

int poisson_likelihood(double *out, double *grad, float *x, unsigned *ij, size_t *dims, unsigned *I0, float *bgd,
                       float *xtal_bi, float *rp, unsigned *fidxs, size_t fsize, unsigned *idxs, size_t isize,
                       unsigned *hkl_idxs, size_t hkl_size, unsigned *oidxs, size_t osize, unsigned threads);

int least_squares(double *out, double *grad, float *x, unsigned *ij, size_t *dims, unsigned *I0, float *bgd, float *xtal_bi,
                  float *rp, unsigned *fidxs, size_t fsize, unsigned *idxs, size_t isize, unsigned *hkl_idxs, size_t hkl_size,
                  unsigned *oidxs, size_t osize, float (*loss_func)(float), float (*grad_func)(float), unsigned threads);

int unmerge_sgn(float *I_hat, float *x, unsigned *ij, size_t *dims, unsigned *I0, float *bgd, float *xtal_bi, float *rp,
                unsigned *fidxs, size_t fsize, unsigned *idxs, size_t isize, unsigned *hkl_idxs, size_t hkl_size,
                unsigned threads);

#endif