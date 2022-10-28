#include "sgn_proc.h"
#include "array.h"

double rbf(double dist, double sigma)
{
    return exp(-0.5 * SQ(dist) / SQ(sigma)) * M_1_SQRT2PI;
}

static void update_window(double *x, size_t *window, size_t *wsize, double x_left, double x_right)
{
    size_t left_end = searchsorted_r(&x_left, window, *wsize, sizeof(size_t), SEARCH_LEFT, indirect_search_double, x);
    size_t right_end = searchsorted_r(&x_right, window, *wsize, sizeof(size_t), SEARCH_LEFT, indirect_search_double, x);
    if (right_end > left_end)
    {
        *wsize = right_end - left_end;
        memmove(window, window + left_end, *wsize * sizeof(size_t));
        window = REALLOC(window, size_t, *wsize);
    }
    else {DEALLOC(window); *wsize = 0; }
}

static double calculate_weights(size_t *window, size_t wsize, size_t ndim, double *y, double *w, double *x, double *xval,
                                kernel krn, double sigma)
{
    int i, j;
    double dist, rbf, Y = 0.0, W = 0.0;

    for (i = 0; i < (int)wsize; i++)
    {
        dist = 0.0;
        for (j = 0; j < (int)ndim; j++) dist += SQ(x[window[i] + j] - xval[j]);
        rbf = krn(sqrt(dist), sigma);
        Y += y[window[i] / ndim] * w[window[i] / ndim] * rbf;
        W += w[window[i] / ndim] * w[window[i] / ndim] * rbf;
    }
    return (W > 0.0) ? Y / W : 0.0;
}

int predict_kerreg(double *y, double *w, double *x, size_t npts, size_t ndim, double *y_hat, double *x_hat, size_t nhat,
                   kernel krn, double sigma, double cutoff, unsigned threads)
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
            size_t left_end = searchsorted_r(&x_left, idxs, npts, sizeof(size_t), SEARCH_LEFT, indirect_search_double, x);
            size_t right_end = searchsorted_r(&x_right, idxs, npts, sizeof(size_t), SEARCH_LEFT, indirect_search_double, x);

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
                y_hat[i] = calculate_weights(window, wsize, ndim, y, w, x, xval->data, krn, sigma);

                DEALLOC(window); wsize = 0;
            }
            else y_hat[i] = 0.0;
        }

        DEALLOC(xval);
    }

    DEALLOC(idxs); free_array(xarr);

    return 0;
}

typedef struct rect_iter_s
{
    int ndim;
    size_t size;
    int *coord;
    size_t *strides;
    int index;
} rect_iter_s;
typedef struct rect_iter_s *rect_iter;

static void ri_del(rect_iter ri)
{
    if (ri == NULL) ERROR("ri_del: NULL iterator.");
    DEALLOC(ri->coord); DEALLOC(ri->strides); DEALLOC(ri);
}

static rect_iter ri_ini(int ndim, int *pt0, int *pt1)
{
    /* check parameters */
    if(ndim <= 0) {ERROR("new_ri: ndim must be positive."); return NULL;}

    rect_iter ri = (rect_iter)malloc(sizeof(struct rect_iter_s));
    if (!ri) {ERROR("new_ri: not enough memory."); return NULL;}
    
    ri->index = 0;
    ri->ndim = ndim;
    ri->size = 1;
    ri->coord = calloc(ri->ndim, sizeof(int));
    ri->strides = MALLOC(size_t, ri->ndim);

    for (int n = ri->ndim - 1; n >= 0; n--)
    {
        if (pt1[n] < pt0[n])
        {
            ERROR("new_ri: pt1 must be larger than pt0");
            ri_del(ri); return NULL;
        }
        ri->strides[n] = ri->size;
        ri->size *= pt1[n] - pt0[n] + 1;
    }

    return ri;
}

static int ri_end(rect_iter ri)
{
    return ri->index >= (int)ri->size;
}

static void ri_inc(rect_iter ri)
{
    if (!ri_end(ri)) ri->index++;
    UNRAVEL_INDEX(ri->coord, &ri->index, ri);
}

int predict_grid(double *y, double *w, double *x, size_t npts, size_t ndim, double *y_hat, const size_t *dims, double *step,
                 kernel krn, double sigma, double cutoff, unsigned threads)
{
    /* check parameters */
    if (!y || !w || !x || !y_hat || !step) {ERROR("predict_grid: one of the arguments is NULL."); return -1;}
    if (!ndim || sigma == 0.0 || cutoff == 0.0) {ERROR("predict_grid: one of the arguments is equal to zero."); return -1;}

    size_t size = 1;
    int *cutoffs = MALLOC(int, ndim);
    for (int i = 0; i < (int)ndim; i++)
    {
        size *= dims[i];
        cutoffs[i] = (int)((cutoff * sigma) / step[i]);
    }

    array Iarr = new_array(ndim, dims, sizeof(double), calloc(threads * size, sizeof(double)));
    array Warr = new_array(ndim, dims, sizeof(double), calloc(threads * size, sizeof(double)));
    double *Iptr = (double *)Iarr->data;
    double *Wptr = (double *)Warr->data;

    #pragma omp parallel num_threads(threads)
    {
        int t = omp_get_thread_num(), i, n, idx;
        double dist, rbf;
        int *pt = MALLOC(int, ndim);
        int *pt0 = MALLOC(int, ndim);
        int *pt1 = MALLOC(int, ndim);
        rect_iter ri;

        #pragma omp for
        for (i = 0; i < (int)npts; i++)
        {
            for (n = 0; n < (int)ndim; n++)
            {
                idx = (int)(x[i * ndim + n] / step[n]) + 1;
                pt0[n] = idx - cutoffs[n]; CLIP(pt0[n], 0, (int)dims[n] - 1);
                pt1[n] = idx + cutoffs[n]; CLIP(pt1[n], 0, (int)dims[n] - 1);
            }

            for (ri = ri_ini(ndim, pt0, pt1); !ri_end(ri); ri_inc(ri))
            {
                for (n = 0; n < ri->ndim; n++) pt[n] = ri->coord[n] + pt0[n];
                RAVEL_INDEX(pt, &idx, Iarr);

                dist = 0.0;
                for (n = 0; n < ri->ndim; n++) dist += SQ(pt[n] * step[n] - x[i * ndim + n]);
                rbf = krn(sqrt(dist), sigma);

                Iptr[t * size + idx] += y[i] * w[i] * rbf;
                Wptr[t * size + idx] += w[i] * w[i] * rbf;
            }
            ri_del(ri);
        }

        DEALLOC(pt); DEALLOC(pt0); DEALLOC(pt1);

        double Ival, Wval;

        #pragma omp for
        for (i = 0; i < (int)size; i++)
        {
            Ival = 0.0; Wval = 0.0;
            for (t = 0; t < (int)threads; t++)
            {
                Ival += Iptr[t * size + i]; Wval += Wptr[t * size + i];
            }
            y_hat[i] = (Wval > 0.0) ? Ival / Wval : 0.0;
        }
    }

    DEALLOC(Iptr); free_array(Iarr);
    DEALLOC(Wptr); free_array(Warr);

    return 0;
}

int unique_indices(unsigned **funiq, unsigned **fidxs, size_t *fpts, unsigned **iidxs, size_t *ipts, unsigned *frames,
                   unsigned *indices, size_t npts)
{
    /* check parameters */
    if (!frames || !indices) {ERROR("unique_indices: one of the arguments is NULL."); return -1;}
    if (npts == 0) {*funiq = NULL; *fidxs = NULL; *iidxs = NULL; *fpts = 0; *ipts = 0; return 0;}

    (*fidxs) = MALLOC(unsigned, frames[npts - 1] - frames[0] + 2);
    (*funiq) = MALLOC(unsigned, frames[npts - 1] - frames[0] + 1);

    // Find 'frames' indices
    unsigned i, j; (*fpts) = 0; (*fidxs)[0] = 0;
    for (i = frames[0]; i <= frames[npts - 1]; i++)
    {
        (*fidxs)[(*fpts) + 1] = searchsorted(&i, frames, npts, sizeof(unsigned), SEARCH_LEFT, compare_uint);
        if ((*fidxs)[(*fpts) + 1] > (*fidxs)[*fpts])
        {
            (*funiq)[*fpts] = i - 1;
            (*fpts)++;
        }
    }
    (*funiq)[(*fpts)] = frames[npts - 1];
    (*fidxs)[++(*fpts)] = npts;

    // Reallocate the buffers
    (*fidxs) = realloc(*fidxs, (*fpts + 1) * sizeof(unsigned));
    (*funiq) = realloc(*funiq, (*fpts) * sizeof(unsigned));

    // Find 'indices' indices
    *ipts = 0;
    *iidxs = MALLOC(unsigned, 1);
    for (i = 0; i < (*fpts); i++)
    {
        (*iidxs) = realloc((*iidxs), ((*ipts) + indices[(*fidxs)[i + 1] - 1] + 1) * sizeof(unsigned));
        for (j = 0; j <= indices[(*fidxs)[i + 1] - 1]; j++)
        {
            (*iidxs)[(*ipts) + j] = (*fidxs)[i] + searchsorted(&j, indices + (*fidxs)[i], (*fidxs)[i + 1] - (*fidxs)[i],
                                                               sizeof(unsigned), SEARCH_LEFT, compare_uint);
        }
        (*ipts) += indices[(*fidxs)[i + 1] - 1] + 1;
    }
    (*iidxs) = realloc((*iidxs), ++(*ipts) * sizeof(unsigned));
    (*iidxs)[(*ipts) - 1] = npts;

    return 0;
}

int update_sf(float *sf, float *sgn, unsigned *xidx, float *xmap, float *xtal, const size_t *ddims,
              unsigned *hkl_idxs, size_t hkl_size, unsigned *iidxs, size_t isize, unsigned threads)
{
    /* check parameters */
    if (!sf || !xidx || !xmap || !xtal || !hkl_idxs || !iidxs)
    {ERROR("update_sf: one of the arguments is NULL."); return -1;}
    if (hkl_size == 0 || isize == 0) {return 0;}

    float *Ibuf = MALLOC(float, isize);
    float *Wbuf = MALLOC(float, isize);

    float *Isum = calloc(hkl_size, sizeof(float));
    float *Wsum = calloc(hkl_size, sizeof(float));

    #pragma omp parallel num_threads(threads)
    {
        int i, j, x0, x1, y0, y1;
        float dx, dy, d_bi, F;

        #pragma omp for
        for (i = 0; i < (int)isize; i++)
        {
            Ibuf[i] = 0.0f; Wbuf[i] = 0.0f;
            for (j = iidxs[i]; j < (int)iidxs[i + 1]; j++)
            {
                if (xmap[2 * j] <= 0.0f)
                {
                    dx = 0.0f; x0 = 0; x1 = 0;
                }
                else if (xmap[2 * j] >= ddims[1] - 1.0f)
                {
                    dx = 0.0f; x0 = ddims[1] - 1; x1 = ddims[1] - 1;
                }
                else
                {
                    dx = xmap[2 * j] - floorf(xmap[2 * j]);
                    x0 = (int)floorf(xmap[2 * j]); x1 = x0 + 1;
                }

                if (xmap[2 * j + 1] <= 0.0f)
                {
                    dy = 0.0f; y0 = 0; y1 = 0;
                }
                else if (xmap[2 * j + 1] >= ddims[2] - 1.0f)
                {
                    dy = 0.0f; y0 = ddims[2] - 1; y1 = ddims[2] - 1;
                }
                else
                {
                    dy = xmap[2 * j + 1] - floorf(xmap[2 * j + 1]);
                    y0 = (int)floorf(xmap[2 * j + 1]); y1 = y0 + 1;
                }

                // Calculate bilinear interpolation
                d_bi = (1.0f - dx) * (1.0f - dy) * xtal[xidx[j] * ddims[1] * ddims[2] + x0 * ddims[2] + y0] +
                               dx  * (1.0f - dy) * xtal[xidx[j] * ddims[1] * ddims[2] + x1 * ddims[2] + y0] +
                       (1.0f - dx) *         dy  * xtal[xidx[j] * ddims[1] * ddims[2] + x0 * ddims[2] + y1] +
                               dx  *         dy  * xtal[xidx[j] * ddims[1] * ddims[2] + x1 * ddims[2] + y1];

                // Calculate weighted least squares slope betta: sgn = betta * d_bi
                Ibuf[i] += sgn[j] * d_bi; Wbuf[i] += SQ(d_bi);
            }
        }

        #pragma omp for
        for (i = 0; i < (int)isize; i++)
        {
            #pragma omp atomic
            Isum[hkl_idxs[i]] += Ibuf[i];
            #pragma omp atomic
            Wsum[hkl_idxs[i]] += Wbuf[i];
        }

        #pragma omp for
        for (i = 0; i < (int)isize; i++)
        {
            F = (Wsum[hkl_idxs[i]] > 0.0f) ? Isum[hkl_idxs[i]] / Wsum[hkl_idxs[i]] : 0.0f;
            for (j = iidxs[i]; j < (int)iidxs[i + 1]; j++) sf[j] = F;
        }
    }

    DEALLOC(Ibuf); DEALLOC(Wbuf);
    DEALLOC(Isum); DEALLOC(Wsum);

    return 0;
}

float scale_crit(float *sf, float *sgn, unsigned *xidx, float *xmap, float *xtal, const size_t *ddims, unsigned *iidxs,
                 size_t isize, unsigned threads)
{
    /* check parameters */
    if (!sf || !xidx || !xmap || !xtal || !iidxs) {ERROR("scale_crit: one of the arguments is NULL."); return 0.0f;}
    if (isize == 0) {return 0.0f;}

    float *Ibuf = MALLOC(float, isize);
    float *Wbuf = MALLOC(float, isize);
    double err = 0.0;

    #pragma omp parallel num_threads(threads) reduction(+:err)
    {
        int i, j, x0, x1, y0, y1;
        float dx, dy, d_bi;

        #pragma omp for
        for (i = 0; i < (int)isize; i++)
        {
            Ibuf[i] = 0.0f; Wbuf[i] = 0.0f;
            for (j = iidxs[i]; j < (int)iidxs[i + 1]; j++)
            {
                if (xmap[2 * j] <= 0.0f)
                {
                    dx = 0.0f; x0 = 0; x1 = 0;
                }
                else if (xmap[2 * j] >= ddims[1] - 1.0f)
                {
                    dx = 0.0f; x0 = ddims[1] - 1; x1 = ddims[1] - 1;
                }
                else
                {
                    dx = xmap[2 * j] - floorf(xmap[2 * j]);
                    x0 = (int)floorf(xmap[2 * j]); x1 = x0 + 1;
                }

                if (xmap[2 * j + 1] <= 0.0f)
                {
                    dy = 0.0f; y0 = 0; y1 = 0;
                }
                else if (xmap[2 * j + 1] >= ddims[2] - 1.0f)
                {
                    dy = 0.0f; y0 = ddims[2] - 1; y1 = ddims[2] - 1;
                }
                else
                {
                    dy = xmap[2 * j + 1] - floorf(xmap[2 * j + 1]);
                    y0 = (int)floorf(xmap[2 * j + 1]); y1 = y0 + 1;
                }

                // Calculate bilinear interpolation
                d_bi = (1.0f - dx) * (1.0f - dy) * xtal[xidx[j] * ddims[1] * ddims[2] + x0 * ddims[2] + y0] +
                               dx  * (1.0f - dy) * xtal[xidx[j] * ddims[1] * ddims[2] + x1 * ddims[2] + y0] +
                       (1.0f - dx) *         dy  * xtal[xidx[j] * ddims[1] * ddims[2] + x0 * ddims[2] + y1] +
                               dx  *         dy  * xtal[xidx[j] * ddims[1] * ddims[2] + x1 * ddims[2] + y1];

                // Calculate weighted least squares slope betta: sgn = betta * d_bi
                err += fabsf(sgn[j] - d_bi * sf[j]);
            }
        }
    }

    DEALLOC(Ibuf); DEALLOC(Wbuf);

    return err / iidxs[isize];
}