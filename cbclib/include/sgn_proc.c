#include "sgn_proc.h"
#include "array.h"

/* find idx \el [0, npts - 1], so that base[idx] <= key < base[idx + 1] */

static size_t search_lbound(const void *key, const void *base, size_t npts, size_t size,
                            int (*compar)(const void *, const void *))
{
    if (compar(key, base) <= 0) return 0;
    if (compar(key, base + (npts - 1) * size) >= 0) return npts - 1;
    return searchsorted(key, base, npts, size, SEARCH_RIGHT, compar) - 1;
}

/* find idx \el [0, npts - 1], so that base[idx - 1] < key <= base[idx] */

static size_t search_rbound(const void *key, const void *base, size_t npts, size_t size,
                            int (*compar)(const void *, const void *))
{
    if (compar(key, base) <= 0) return 0;
    if (compar(key, base + (npts - 1) * size) >= 0) return npts - 1;
    return searchsorted(key, base, npts, size, SEARCH_RIGHT, compar);
}

static size_t search_lbound_r(const void *key, const void *base, size_t npts, size_t size,
                              int (*compar)(const void *, const void *, void *), void *arg)
{
    if (compar(key, base, arg) <= 0) return 0;
    if (compar(key, base + (npts - 1) * size, arg) >= 0) return npts - 1;
    return searchsorted_r(key, base, npts, size, SEARCH_RIGHT, compar, arg) - 1;
}

static size_t search_rbound_r(const void *key, const void *base, size_t npts, size_t size,
                              int (*compar)(const void *, const void *, void *), void *arg)
{
    if (compar(key, base, arg) <= 0) return 0;
    if (compar(key, base + (npts - 1) * size, arg) >= 0) return npts - 1;
    return searchsorted_r(key, base, npts, size, SEARCH_RIGHT, compar, arg);
}

/*----------------------------------------------------------------------------*/
/*------------------------- Bilinear interpolation ---------------------------*/
/*----------------------------------------------------------------------------*/

/* points (integer) follow the convention:      [..., k, j, i], where {i <-> x, j <-> y, k <-> z}
   coordinates (float) follow the convention:   [x, y, z, ...]
 */

static float bilinearf(array obj, float **grid, float *crd)
{
    int i, n, idx;
    float cf, v_bi = 0.0f;

    float *dx = MALLOC(float, obj->ndim);
    int *pt0 = MALLOC(int, obj->ndim);
    int *pt1 = MALLOC(int, obj->ndim);
    int *pt = MALLOC(int, obj->ndim);

    for (n = 0; n < obj->ndim; n++)
    {
        pt0[obj->ndim - 1 - n] = search_lbound(crd + n, grid[n], obj->dims[obj->ndim - 1 - n], sizeof(float), compare_float);
        pt1[obj->ndim - 1 - n] = search_rbound(crd + n, grid[n], obj->dims[obj->ndim - 1 - n], sizeof(float), compare_float);
        if (pt0[obj->ndim - 1 - n] != pt1[obj->ndim - 1 - n])
        {
            dx[n] = (crd[n] - grid[n][pt0[obj->ndim - 1 - n]]) / (grid[n][pt1[obj->ndim - 1 - n]] - grid[n][pt0[obj->ndim - 1 - n]]);
        }
        else dx[n] = 0.0f;
    }

    int size = 1 << n;
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

int interp_bi(float *out, float *data, int ndim, const size_t *dims, float **grid, float *crds, size_t ncrd,
              unsigned threads)
{
    if (!data || !dims || !crds) {ERROR("interp_bi: one of the arguments is NULL."); return -1;}
    if (ndim == 0) {ERROR("interp_bi: ndim must be a positive number"); return -1;}

    if (ncrd == 0) return 0;

    array obj = new_array(ndim, dims, sizeof(float), data);

    #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < (int)ncrd; i++)
    {
        out[i] = bilinearf(obj, grid, crds + ndim * i);
    }

    free_array(obj);

    return 0;
}

/*----------------------------------------------------------------------------*/
/*---------------------------- Kernel regression -----------------------------*/
/*----------------------------------------------------------------------------*/

#define CUTOFF 4.0

static void update_window(float *x, size_t *window, size_t *wsize, float x_left, float x_right)
{
    size_t left_end = search_lbound_r(&x_left, window, *wsize, sizeof(size_t), indirect_search_float, x);
    size_t right_end = search_rbound_r(&x_right, window, *wsize, sizeof(size_t), SEARCH_LEFT, x) + 1;
    if (right_end > left_end)
    {
        *wsize = right_end - left_end;
        memmove(window, window + left_end, *wsize * sizeof(size_t));
        window = REALLOC(window, size_t, *wsize);
    }
    else {DEALLOC(window); *wsize = 0; }
}

static float calculate_weights(size_t *window, size_t wsize, size_t ndim, float *y, float *w, float *x, float *xval,
                                kernel krn, float sigma)
{
    int i, j;
    float dist, rbf, Y = 0.0, W = 0.0;

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

int predict_kerreg(float *y, float *w, float *x, size_t npts, size_t ndim, float *y_hat, float *x_hat, size_t nhat,
                   kernel krn, float sigma, unsigned threads)
{
    /* check parameters */
    if (!y || !x || !y_hat || !x_hat) {ERROR("predict_kerreg: one of the arguments is NULL."); return -1;}
    if (!ndim || sigma == 0.0) {ERROR("predict_kerreg: one of the arguments is equal to zero."); return -1;}

    int i;
    size_t *idxs = MALLOC(size_t, npts);

    for (i = 0; i < (int)npts; i++) idxs[i] = ndim * i;
    POSIX_QSORT_R(idxs, npts, sizeof(size_t), indirect_compare_float, (void *)x);

    size_t xsize[2] = {nhat, ndim};
    array xarr = new_array(2, xsize, sizeof(float), x_hat);

    int repeats = xarr->size / xarr->dims[1];
    threads = (threads > (unsigned) repeats) ? (unsigned) repeats : threads;

    #pragma omp parallel num_threads(threads)
    {
        int j, axis;
        float x_left, x_right;
        size_t wsize, *window;

        line xval = init_line(xarr, 1);

        #pragma omp for
        for (i = 0; i < (int)repeats; i++)
        {
            UPDATE_LINE(xval, i);

            axis = 0;
            x_left = GET(xval, float, axis) - CUTOFF * sigma;
            x_right = GET(xval, float, axis) + CUTOFF * sigma;
            size_t left_end = search_lbound_r(&x_left, idxs, npts, sizeof(size_t), indirect_search_float, x);
            size_t right_end = search_rbound_r(&x_right, idxs, npts, sizeof(size_t), indirect_search_float, x) + 1;

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
                POSIX_QSORT_R(window, wsize, sizeof(size_t), indirect_compare_float, (void *)x);
                x_left = GET(xval, float, axis) - CUTOFF * sigma;
                x_right = GET(xval, float, axis) + CUTOFF * sigma;
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
int predict_grid(float **y_hat, size_t *roi, float *y, float *w, float *x, size_t npts, size_t ndim, float **grid,
                 const size_t *gdims, kernel krn, float sigma, unsigned threads)
{
    /* check parameters */
    if (!y || !w || !x || !y_hat || !grid) {ERROR("predict_grid: one of the arguments is NULL."); return -1;}
    if (!ndim || sigma == 0.0) {ERROR("predict_grid: one of the arguments is equal to zero."); return -1;}

    size_t size = 1;
    size_t *dims = MALLOC(size_t, ndim);
    for (int n = 0; n < (int)ndim; n++)
    {
        float x_min = FLT_MAX, x_max = FLT_MIN;
        for (int i = 0; i < (int)npts; i++)
        {
            if (x[ndim * i + ndim - 1 - n] < x_min) x_min = x[ndim * i + ndim - 1 - n];
            if (x[ndim * i + ndim - 1 - n] > x_max) x_max = x[ndim * i + ndim - 1 - n];
        }
        roi[2 * n] = search_lbound(&x_min, grid[ndim - 1 - n], gdims[n], sizeof(float), compare_float);
        roi[2 * n + 1] = search_rbound(&x_max, grid[ndim - 1 - n], gdims[n], sizeof(float), compare_float) + 1;
        dims[n] = roi[2 * n + 1] - roi[2 * n];
        size *= dims[n];
    }

    (*y_hat) = MALLOC(float, size);
    array Iarr = new_array(ndim, dims, sizeof(float), calloc(threads * size, sizeof(float)));
    array Warr = new_array(ndim, dims, sizeof(float), calloc(threads * size, sizeof(float)));

    #pragma omp parallel num_threads(threads)
    {
        int t = omp_get_thread_num(), i, n, idx;
        float dist, rbf, crd;
        int *pt = MALLOC(int, ndim);
        int *pt0 = MALLOC(int, ndim);
        int *pt1 = MALLOC(int, ndim);
        rect_iter ri;

        #pragma omp for
        for (i = 0; i < (int)npts; i++)
        {
            for (n = 0; n < (int)ndim; n++)
            {
                crd = x[i * ndim + ndim - 1 - n] - CUTOFF * sigma;
                pt0[n] = search_lbound(&crd, grid[ndim - 1 - n], gdims[n], sizeof(float), compare_float);
                CLIP(pt0[n], (int)roi[2 * n], (int)roi[2 * n + 1]);
                crd = x[i * ndim + ndim - 1 - n] + CUTOFF * sigma;
                pt1[n] = search_rbound(&crd, grid[ndim - 1 - n], gdims[n], sizeof(float), compare_float) + 1;
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
                    dist += SQ(grid[ndim - 1 - n][pt[n] + roi[2 * n]] - x[i * ndim + ndim - 1 - n]);
                }
                rbf = krn(sqrt(dist), sigma);

                GET(Iarr, float, t * size + idx) += y[i] * w[i] * rbf;
                GET(Warr, float, t * size + idx) += w[i] * w[i] * rbf;
            }
            ri_del(ri);
        }

        DEALLOC(pt); DEALLOC(pt0); DEALLOC(pt1);

        float Ival, Wval;

        #pragma omp for
        for (i = 0; i < (int)size; i++)
        {
            Ival = 0.0; Wval = 0.0;
            for (t = 0; t < (int)threads; t++)
            {
                Ival += GET(Iarr, float, t * size + i);
                Wval += GET(Warr, float, t * size + i);
            }
            (*y_hat)[i] = (Wval > 0.0) ? Ival / Wval : 0.0;
        }
    }

    DEALLOC(Iarr->data); free_array(Iarr);
    DEALLOC(Warr->data); free_array(Warr);
    DEALLOC(dims);

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
    (*fidxs) = REALLOC(*fidxs, unsigned, *fpts + 1);
    (*funiq) = REALLOC(*funiq, unsigned, *fpts);

    // Find 'indices' indices
    *ipts = 0;
    *iidxs = MALLOC(unsigned, 1);
    for (i = 0; i < (*fpts); i++)
    {
        (*iidxs) = REALLOC(*iidxs, unsigned, (*ipts) + indices[(*fidxs)[i + 1] - 1] + 1);
        for (j = 0; j <= indices[(*fidxs)[i + 1] - 1]; j++)
        {
            (*iidxs)[(*ipts) + j] = (*fidxs)[i] + searchsorted(&j, indices + (*fidxs)[i], (*fidxs)[i + 1] - (*fidxs)[i],
                                                               sizeof(unsigned), SEARCH_LEFT, compare_uint);
        }
        (*ipts) += indices[(*fidxs)[i + 1] - 1] + 1;
    }
    (*iidxs) = REALLOC(*iidxs, unsigned, ++(*ipts));
    (*iidxs)[(*ipts) - 1] = npts;

    return 0;
}

/*----------------------------------------------------------------------------*/
/*---------------------- Intensity scaling criterions ------------------------*/
/*----------------------------------------------------------------------------*/

float poisson_likelihood(float *grad, float *x, size_t xsize, float *rp, unsigned *I0, float *bgd, float *xtal_bi,
                         unsigned *hkl_idxs, unsigned *iidxs, size_t isize, unsigned threads)
{
    /* Check parameters */
    if (!x || !rp || !I0 || !bgd || !xtal_bi || !hkl_idxs || !iidxs)
    {ERROR("poisson_likelihood: one of the arguments is NULL."); return 0.0f;}

    float criterion = 0.0f;
    if (isize == 0) return criterion;

    #pragma omp parallel num_threads(threads)
    {
        int i, j, idx;
        float crit, gval, y_hat;

        #pragma omp for
        for (i = 0; i < (int)isize; i++)
        {
            crit = gval = 0.0f;
            idx = isize + hkl_idxs[i];
            for (j = iidxs[i]; j < (int)iidxs[i + 1]; j++)
            {
                y_hat = expf(x[idx]) * xtal_bi[j] * rp[j] + x[i] + bgd[j];
                if (y_hat > 0.0f)
                {
                    crit -= I0[j] * logf(y_hat) - y_hat;
                    grad[i] -= I0[j] / y_hat - 1.0f;
                    gval -= I0[j] / y_hat * expf(x[idx]) * xtal_bi[j] * rp[j] - expf(x[idx]) * xtal_bi[j] * rp[j];
                }
            }

            #pragma omp atomic
            grad[idx] += gval;
            #pragma omp atomic
            criterion += crit;
        }
    }

    return criterion;
}

float least_squares(float *grad, float *x, size_t xsize, float *rp, unsigned *I0, float *bgd, float *xtal_bi,
                    unsigned *hkl_idxs, unsigned *iidxs, size_t isize, float (*loss_func)(float), float (*grad_func)(float),
                    unsigned threads)
{
    /* Check parameters */
    if (!x || !rp || !I0 || !bgd || !xtal_bi || !hkl_idxs || !iidxs)
    {ERROR("least_squares: one of the arguments is NULL."); return 0.0f;}

    float criterion = 0.0f;
    if (isize == 0) return criterion;

    #pragma omp parallel num_threads(threads)
    {
        int i, j, idx;
        float crit, gval, std, y_hat;

        #pragma omp for
        for (i = 0; i < (int)isize; i++)
        {
            crit = gval = std = 0.0f;
            idx = isize + hkl_idxs[i];
            for (j = iidxs[i]; j < (int)iidxs[i + 1]; j++) std += I0[j];
            std = sqrtf(std / (iidxs[i + 1] - iidxs[i]));
            for (j = iidxs[i]; j < (int)iidxs[i + 1]; j++)
            {
                y_hat = expf(x[idx]) * xtal_bi[j] * rp[j] + x[i] + bgd[j];
                crit += loss_func((I0[j] - y_hat) / std);
                grad[i] -= grad_func((I0[j] - y_hat) / std) / std;
                gval -= grad_func((I0[j] - y_hat) / std) * expf(x[idx]) * xtal_bi[j] * rp[j] / std;
            }

            #pragma omp atomic
            grad[idx] += gval;
            #pragma omp atomic
            criterion += crit;
        }
    }

    return criterion;
}