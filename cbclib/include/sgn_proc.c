#include "sgn_proc.h"
#include "array.h"

/*----------------------------------------------------------------------------*/
/*------------------------- Bilinear interpolation ---------------------------*/
/*----------------------------------------------------------------------------*/

/* points (integer) follow the convention:      [..., k, j, i], where {i <-> x, j <-> y, k <-> z}
   coordinates (float) follow the convention:   [x, y, z, ...]
 */

static float bilinearf(array obj, float *crd)
{
    int i, n, idx, size = 1;
    float cf, v_bi = 0.0f;

    float *dx = MALLOC(float, obj->ndim);
    int *pt0 = MALLOC(int, obj->ndim);
    int *pt1 = MALLOC(int, obj->ndim);
    int *pt = MALLOC(int, obj->ndim);

    for (n = 0; n < obj->ndim; n++)
    {
        if (crd[n] <= 0.0f)
        {
            dx[n] = 0.0f; pt0[obj->ndim - 1 - n] = pt1[obj->ndim - 1 - n] = 0;
        }
        else if (crd[n] >= obj->dims[obj->ndim - 1 - n] - 1)
        {
            dx[n] = 0.0f;
            pt0[obj->ndim - 1 - n] = obj->dims[obj->ndim - 1 - n] - 1;
            pt1[obj->ndim - 1 - n] = obj->dims[obj->ndim - 1 - n] - 1;
        }
        else
        {
            dx[n] = crd[n] - floorf(crd[n]);
            pt0[obj->ndim - 1 - n] = (int)floorf(crd[n]);
            pt1[obj->ndim - 1 - n] = pt0[obj->ndim - 1 - n] + 1;
        }
    }

    for (n = 0; n < obj->ndim; n++) size <<= 1;
    for (i = 0; i < size; i++)
    {
        cf = 1.0f;
        for (n = 0; n < obj->ndim; n++)
        {
            if ((i >> n) & 1)
            {
                pt[obj->ndim - 1 - n] = pt1[obj->ndim - 1 - n];
                cf *= 1.0f - dx[n];
            }
            else
            {
                pt[obj->ndim - 1 - n] = pt0[obj->ndim - 1 - n];
                cf *= dx[n];
            }
        }
        RAVEL_INDEX(pt, &idx, obj);
        v_bi += cf * GET(obj, float, idx);
    }

    DEALLOC(dx); DEALLOC(pt0); DEALLOC(pt1); DEALLOC(pt);
    return v_bi;
}

int interp_bi(float *out, float *data, int ndim, const size_t *dims, float *crds, size_t ncrd)
{
    if (!data || !dims || !crds) {ERROR("interp_bi: one of the arguments is NULL."); return -1;}
    if (ndim == 0) {ERROR("interp_bi: ndim must be a positive number"); return -1;}

    if (ncrd == 0) return 0;

    int i;
    float *crd;

    array obj = new_array(ndim, dims, sizeof(float), data);

    for (i = 0, crd = crds; i < (int)ncrd; i++, crd += ndim)
    {
        out[i] = bilinearf(obj, crd);
    }

    free_array(obj);

    return 0;
}

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
            x_left = GET(xval, double, axis) - cutoff;
            x_right = GET(xval, double, axis) + cutoff;
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
                x_left = GET(xval, double, axis) - cutoff;
                x_right = GET(xval, double, axis) + cutoff;
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

/* points (integer) follow the convention:      [..., k, j, i], where {i <-> x, j <-> y, k <-> z}
   coordinates (float) follow the convention:   [x, y, z, ...]
 */
int predict_grid(double *y, double *w, double *x, size_t npts, size_t ndim, double *y_hat, const size_t *roi,
                 double *step, kernel krn, double sigma, double cutoff, unsigned threads)
{
    /* check parameters */
    if (!y || !w || !x || !y_hat || !step) {ERROR("predict_grid: one of the arguments is NULL."); return -1;}
    if (!ndim || sigma == 0.0 || cutoff == 0.0) {ERROR("predict_grid: one of the arguments is equal to zero."); return -1;}

    size_t size = 1;
    int *cutoffs = MALLOC(int, ndim);
    size_t *dims = MALLOC(size_t, ndim);
    for (int i = 0; i < (int)ndim; i++)
    {
        dims[i] = roi[2 * i + 1] - roi[2 * i];
        size *= dims[i];
        cutoffs[i] = (int)((cutoff * sigma) / step[ndim - 1 - i]);
    }

    array Iarr = new_array(ndim, dims, sizeof(double), calloc(threads * size, sizeof(double)));
    array Warr = new_array(ndim, dims, sizeof(double), calloc(threads * size, sizeof(double)));

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
                idx = (int)ceilf(x[i * ndim + ndim - 1 - n] / step[ndim - 1 - n]);
                pt0[n] = idx - cutoffs[n];
                CLIP(pt0[n], (int)roi[2 * n], (int)roi[2 * n + 1]);
                pt1[n] = idx + cutoffs[n];
                CLIP(pt1[n], (int)roi[2 * n], (int)roi[2 * n + 1]);
            }

            for (ri = ri_ini(ndim, pt0, pt1); !ri_end(ri); ri_inc(ri))
            {
                // Calculate the coordinate of the buffer array
                for (n = 0; n < ri->ndim; n++) pt[n] = ri->coord[n] + pt0[n] - roi[2 * n];
                RAVEL_INDEX(pt, &idx, Iarr);

                // Calculate the distance
                dist = 0.0;
                for (n = 0; n < ri->ndim; n++)
                {
                    dist += SQ((pt[n] + roi[2 * n]) * step[ndim - 1 - n] - x[i * ndim + ndim - 1 - n]);
                }
                rbf = krn(sqrt(dist), sigma);

                GET(Iarr, double, t * size + idx) += y[i] * w[i] * rbf;
                GET(Warr, double, t * size + idx) += w[i] * w[i] * rbf;
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
                Ival += GET(Iarr, double, t * size + i);
                Wval += GET(Warr, double, t * size + i);
            }
            y_hat[i] = (Wval > 0.0) ? Ival / Wval : 0.0;
        }
    }

    DEALLOC(Iarr->data); free_array(Iarr);
    DEALLOC(Warr->data); free_array(Warr);
    DEALLOC(cutoffs); DEALLOC(dims);

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

// static float xtal_bilinear(unsigned xidx, float xmap_x, float xmap_y, float *xtal, const size_t *ddims)
// {
//     float dx, dy, xtal_bi;
//     int x0, x1, y0, y1;
//     if (xmap_x <= 0.0f)
//     {
//         dx = 0.0f; x0 = 0; x1 = 0;
//     }
//     else if (xmap_x >= ddims[2] - 1.0f)
//     {
//         dx = 0.0f; x0 = ddims[2] - 1; x1 = ddims[2] - 1;
//     }
//     else
//     {
//         dx = xmap_x - floorf(xmap_x);
//         x0 = (int)floorf(xmap_x); x1 = x0 + 1;
//     }

//     if (xmap_y <= 0.0f)
//     {
//         dy = 0.0f; y0 = 0; y1 = 0;
//     }
//     else if (xmap_y >= ddims[1] - 1.0f)
//     {
//         dy = 0.0f; y0 = ddims[1] - 1; y1 = ddims[1] - 1;
//     }
//     else
//     {
//         dy = xmap_y - floorf(xmap_y);
//         y0 = (int)floorf(xmap_y); y1 = y0 + 1;
//     }

//     // Calculate bilinear interpolation
//     xtal_bi = (1.0f - dx) * (1.0f - dy) * xtal[xidx * ddims[1] * ddims[2] + y0 * ddims[2] + x0] +
//                       dx  * (1.0f - dy) * xtal[xidx * ddims[1] * ddims[2] + y1 * ddims[2] + x0] +
//               (1.0f - dx) *         dy  * xtal[xidx * ddims[1] * ddims[2] + y0 * ddims[2] + x1] +
//                       dx  *         dy  * xtal[xidx * ddims[1] * ddims[2] + y1 * ddims[2] + x1];
//     return xtal_bi;
// }

int update_sf(float *sf, float *dsf, float *rp, float *sgn, unsigned *xidx, float *xmap, float *xtal, const size_t *ddims,
              unsigned *hkl_idxs, size_t hkl_size, unsigned *iidxs, size_t isize, unsigned threads)
{
    /* Check parameters */
    if (!sf || !dsf || !rp || !sgn || !xidx || !xmap || !xtal || !hkl_idxs || !iidxs)
    {ERROR("update_sf: one of the arguments is NULL."); return -1;}
    if (hkl_size == 0 || isize == 0) return 0;

    float *Ibuf = MALLOC(float, isize);
    float *dIbuf = MALLOC(float, isize);
    float *Wbuf = MALLOC(float, isize);

    float *Isum = calloc(hkl_size, sizeof(float));
    float *dIsum = calloc(hkl_size, sizeof(float));
    float *Wsum = calloc(hkl_size, sizeof(float));

    array xarr = new_array(3, ddims, sizeof(float), xtal);

    #pragma omp parallel num_threads(threads)
    {
        int i, j;
        float xtal_bi, F, dF, crd[3];

        #pragma omp for
        for (i = 0; i < (int)isize; i++)
        {
            Ibuf[i] = 0.0f; Wbuf[i] = 0.0f; dIbuf[i] = 0.0f;
            for (j = iidxs[i]; j < (int)iidxs[i + 1]; j++)
            {
                /* Assign a coordinate */
                crd[0] = xmap[2 * j]; crd[1] = xmap[2 * j + 1]; crd[2] = xidx[j];
                // Calculate xtal at a given position xmap using the bilinear interpolation
                xtal_bi = bilinearf(xarr, crd);

                /* Calculate weighted least squares slope betta: sgn = betta * xtal_bi * rp
                   The slope is given by:
                   betta = \sum_{j \el (h, k, l)} (sgn[j] * xtal[j] * rp[j]) *
                           \sum_{j \el (h, k, l)} (xtal[j]^2 * rp[j]^2)
                 */
                Ibuf[i] += sgn[j] * xtal_bi * rp[j];
                Wbuf[i] += SQ(xtal_bi * rp[j]);
                dIbuf[i] += fabsf(sgn[j]) * SQ(xtal_bi * rp[j]);
            }
        }

        #pragma omp for
        for (i = 0; i < (int)isize; i++)
        {
            #pragma omp atomic
            Isum[hkl_idxs[i]] += Ibuf[i];
            #pragma omp atomic
            dIsum[hkl_idxs[i]] += dIbuf[i];
            #pragma omp atomic
            Wsum[hkl_idxs[i]] += Wbuf[i];
        }

        #pragma omp for
        for (i = 0; i < (int)isize; i++)
        {
            F = (Wsum[hkl_idxs[i]] > 0.0f) ? Isum[hkl_idxs[i]] / Wsum[hkl_idxs[i]] : 0.0f;
            dF = (Wsum[hkl_idxs[i]] > 0.0f) ? sqrtf(dIsum[hkl_idxs[i]]) / Wsum[hkl_idxs[i]] : 0.0f;
            for (j = iidxs[i]; j < (int)iidxs[i + 1]; j++) {sf[j] = F; dsf[j] = dF;}
        }
    }

    DEALLOC(Ibuf); DEALLOC(Wbuf); DEALLOC(dIbuf);
    DEALLOC(Isum); DEALLOC(Wsum); DEALLOC(dIsum);

    free_array(xarr);

    return 0;
}

float scale_crit(float *sf, float *rp, float *sgn, unsigned *xidx, float *xmap, float *xtal, const size_t *ddims,
                 unsigned *iidxs, size_t isize, unsigned threads)
{
    /* Check parameters */
    if (!sf || !rp || !sgn || !xidx || !xmap || !xtal || !iidxs)
    {ERROR("scale_crit: one of the arguments is NULL."); return 0.0f;}
    if (isize == 0) return 0.0f;

    double err = 0.0;
    array xarr = new_array(3, ddims, sizeof(float), xmap);

    #pragma omp parallel num_threads(threads) reduction(+:err)
    {
        int i, j;
        float xtal_bi, crd[3];

        #pragma omp for
        for (i = 0; i < (int)isize; i++)
        {
            for (j = iidxs[i]; j < (int)iidxs[i + 1]; j++)
            {
                // Assign a coordinate
                crd[0] = xmap[2 * j]; crd[1] = xmap[2 * j + 1]; crd[2] = xidx[j];
                // Calculate xtal at a given position xmap
                xtal_bi = bilinearf(xarr, crd);

                // Calculate weighted least squares slope betta: sgn = betta * d_bi
                err += fabsf(sgn[j] - xtal_bi * rp[j] * sf[j]);
            }
        }
    }

    free_array(xarr);

    return err / iidxs[isize];
}

int xtal_interp(float *xtal_bi, unsigned *xidx, float *xmap, float *xtal, const size_t *ddims, size_t isize, unsigned threads)
{
    /* check parameters */
    if (!xtal_bi || !xidx || !xmap || !xtal) {ERROR("xtal_interp: one of the arguments is NULL."); return -1;}
    if (isize == 0) return 0;

    array xarr = new_array(3, ddims, sizeof(float), xtal);

    #pragma omp parallel num_threads(threads)
    {
        float crd[3];

        #pragma omp for
        for (int i = 0; i < (int)isize; i++)
        {
            // Assign a coordinate
            crd[0] = xmap[2 * i]; crd[1] = xmap[2 * i + 1]; crd[2] = xidx[i];
            // Calculate xtal at a given position xmap
            xtal_bi[i] = bilinearf(xarr, crd);
        }
    }

    free_array(xarr);

    return 0;
}