#include "smoothers.h"
#include "array.h"

double rbf(double dist, double sigma)
{
    return exp(-0.5 * SQ(dist) / SQ(sigma)) * M_1_SQRT2PI;
}

static void update_window(double *x, size_t *window, size_t *wsize, double x_left, double x_right)
{
    size_t left_end = searchsorted_r(&x_left, window, *wsize, sizeof(size_t), indirect_search_double, x);
    size_t right_end = searchsorted_r(&x_right, window, *wsize, sizeof(size_t), indirect_search_double, x);
    if (right_end > left_end)
    {
        *wsize = right_end - left_end;
        memmove(window, window + left_end, *wsize * sizeof(size_t));
        window = REALLOC(window, size_t, *wsize);
    }
    else {DEALLOC(window); *wsize = 0; }
}

static double calculate_weights(size_t *window, size_t wsize, size_t ndim, double *y, double *x, double *xval,
                                kernel krn, double sigma, double epsilon)
{
    int i, j;
    double dist, w, Y = 0.0, W = 0.0;

    for (i = 0; i < (int)wsize; i++)
    {
        dist = 0.0;
        for (j = 0; j < (int)ndim; j++) dist += SQ(x[window[i] + j] - xval[j]);
        w = krn(sqrt(dist), sigma); W += w; Y += y[window[i] / ndim] * w;
    }
    return Y / (W + epsilon);
}

int predict_kerreg(double *y, double *x, size_t npts, size_t ndim, double *y_hat, double *x_hat, size_t nhat,
                   kernel krn, double sigma, double cutoff, double epsilon, unsigned threads)
{
    /* check parameters */
    if (!y || !x || !y_hat || !x_hat) {ERROR("predict_kerreg: one of the arguments is NULL."); return -1;}
    if (!ndim || (sigma == 0.0) || (cutoff == 0.0)) {ERROR("predict_kerreg: one of the arguments is equal to zero."); return -1;}

    int i;
    size_t *idxs = MALLOC(size_t, npts);

    for (i = 0; i < (int)npts; i++) idxs[i] = ndim * i;
    POSIX_QSORT_R(idxs, npts, sizeof(size_t), indirect_compare_double, (void *)x);

    size_t xsize[2] = {nhat, ndim};
    array xarr = new_array(2, xsize, sizeof(double), x_hat);

    int repeats = xarr->size / xarr->dims[1];
    threads = (threads > (unsigned) repeats) ? (unsigned) repeats : threads;

    #pragma omp parallel num_threads(threads)
    {
        int j, axis;
        double x_left, x_right;
        size_t wsize, *window;

        line xval = init_line(xarr, 1);

        #pragma omp for
        for (i = 0; i < (int)repeats; i++)
        {
            UPDATE_LINE(xval, i);

            axis = 0;
            x_left = ((double *)xval->data)[axis] - cutoff;
            x_right = ((double *)xval->data)[axis] + cutoff;
            size_t left_end = searchsorted_r(&x_left, idxs, npts, sizeof(size_t), indirect_search_double, x);
            size_t right_end = searchsorted_r(&x_right, idxs, npts, sizeof(size_t), indirect_search_double, x);

            if (right_end > left_end)
            {
                wsize = right_end - left_end;
                window = MALLOC(size_t, wsize);
                memmove(window, idxs + left_end, wsize * sizeof(size_t));
            }

            y_hat[i] = 0.0;

            while (wsize && (++axis < (int)ndim))
            {
                for (j = 0; j < (int)wsize; j++) window[j]++;
                POSIX_QSORT_R(window, wsize, sizeof(size_t), indirect_compare_double, (void *)x);
                x_left = ((double *)xval->data)[axis] - cutoff;
                x_right = ((double *)xval->data)[axis] + cutoff;
                update_window(x, window, &wsize, x_left, x_right);
            }


            if (wsize)
            {
                for (j = 0; j < (int)wsize; j++) window[j] -= ndim - 1;
                y_hat[i] = calculate_weights(window, wsize, ndim, y, x, xval->data, krn, sigma, epsilon);

                DEALLOC(window); wsize = 0;
            }
            else y_hat[i] = 0.0;
        }

        DEALLOC(xval);
    }

    DEALLOC(idxs); free_array(xarr);

    return 0;
}