#ifndef SMOOTHERS_H
#define SMOOTHERS_H
#include "include.h"

typedef double (*kernel)(double dist, double sigma);

double rbf(double dist, double sigma);

/*-------------------------------------------------------------------------------*/
/** Predict a smoothed curve with Nadaraya-Watson kernel regression.

    @param y            Buffer of input y values.
    @param x            Buffer of input x values.
    @param npts         Number of (y, x) points.
    @param ndim         Number of dimensions of x.
    @param yhat         Output buffer, where the predicted curve is written.
    @param xhat         Points, where the prediction is calculated.
    @param nhat         Number of xhat points.
    @param krn          Kernel function krn(dist, sigma) of distance dist and kernel
                        bandwidth sigma.
    @param cutoff       Prediction is calculated based on points from [x - cutoff,
                        x + cutoff] range.
    @param epsilon      Fall-off parameter. Ensures, that prediction goes to 0 outside
                        of the input domain.
    @param threads      Number of threads used during the calculation.

    @return             Returns 0 if it finished normally, 1 otherwise.
 */
int predict_kerreg(double *y, double *x, size_t npts, size_t ndim, double *yhat, double *xhat, size_t nhat,
                   kernel krn, double sigma, double cutoff, double epsilon, unsigned threads);

#endif