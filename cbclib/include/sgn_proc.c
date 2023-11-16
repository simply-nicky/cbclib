#include "sgn_proc.h"
#include "array.h"

/* find idx \el [0, npts - 1], so that base[idx] <= key < base[idx + 1] */

static size_t search_lbound(const void *key, const void *base, size_t npts, size_t size,
                            int (*compar)(const void *, const void *))
{
    size_t idx = searchsorted(key, base, npts, size, SEARCH_RIGHT, compar);
    if (idx) return idx - 1;
    return 0;
}

/* find idx \el [0, npts - 1], so that base[idx - 1] < key <= base[idx] */

static size_t search_rbound(const void *key, const void *base, size_t npts, size_t size,
                            int (*compar)(const void *, const void *))
{
    size_t idx = searchsorted(key, base, npts, size, SEARCH_RIGHT, compar);
    if (idx == npts) return npts ? npts - 1 : 0;
    return idx;
}

static size_t search_lbound_r(const void *key, const void *base, size_t npts, size_t size,
                              int (*compar)(const void *, const void *, void *), void *arg)
{
    size_t idx = searchsorted_r(key, base, npts, size, SEARCH_RIGHT, compar, arg);
    if (idx) return idx - 1;
    return 0;
}

static size_t search_rbound_r(const void *key, const void *base, size_t npts, size_t size,
                              int (*compar)(const void *, const void *, void *), void *arg)
{
    size_t idx = searchsorted_r(key, base, npts, size, SEARCH_RIGHT, compar, arg);
    if (idx == npts) return npts ? npts - 1 : 0;
    return idx;
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
                cf *= dx[n];
            }
            else
            {
                pt[obj->ndim - 1 - n] = pt0[obj->ndim - 1 - n];
                cf *= 1.0f - dx[n];
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

#define CUTOFF 3.0

static void update_window(float *x, size_t *window, size_t *wsize, float x_left, float x_right)
{
    size_t left_end = search_lbound_r(&x_left, window, *wsize, sizeof(size_t), indirect_search_float, x);
    size_t right_end = search_rbound_r(&x_right, window, *wsize, sizeof(size_t), indirect_search_float, x);
    if (right_end != (*wsize)) right_end++;

    *wsize = right_end - left_end;
    if (*wsize)
    {
        memmove(window, window + left_end, *wsize * sizeof(size_t));
        window = REALLOC(window, size_t, *wsize);
    }
    else DEALLOC(window);
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
            size_t right_end = search_rbound_r(&x_right, idxs, npts, sizeof(size_t), indirect_search_float, x);
            if (right_end != npts) right_end++;

            wsize = right_end - left_end;
            if (wsize)
            {
                window = MALLOC(size_t, wsize);
                memmove(window, idxs + left_end, wsize * sizeof(size_t));
            }

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

                DEALLOC(window);
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

int unique_idxs(unsigned **unique, unsigned **iidxs, size_t *isize, unsigned *indices, unsigned *inverse, size_t npts)
{
    /* check parameters */
    if (!unique || !iidxs || !indices) {ERROR("unique_idxs: one of the arguments is NULL."); return -1;}
    if (npts == 0) {*unique = NULL; *iidxs = NULL; *isize = 0; return 0;}

    (*iidxs) = MALLOC(unsigned, indices[npts - 1] - indices[0] + 2);
    (*unique) = MALLOC(unsigned, indices[npts - 1] - indices[0] + 1);

    // Find 'indices' indices
    unsigned i, j; (*isize) = 0; (*iidxs)[0] = 0;
    for (i = indices[0]; i <= indices[npts - 1]; i++)
    {
        (*iidxs)[(*isize) + 1] = searchsorted(&i, indices, npts, sizeof(unsigned), SEARCH_LEFT, compare_uint);
        if ((*iidxs)[(*isize) + 1] > (*iidxs)[*isize])
        {
            (*unique)[*isize] = i - 1;
            for (j = (*iidxs)[*isize]; j < (*iidxs)[(*isize) + 1]; j++) inverse[j] = (*isize);
            (*isize)++;
        }
    }
    (*unique)[(*isize)] = indices[npts - 1];
    for (j = (*iidxs)[*isize]; j < npts; j++) inverse[j] = (*isize);
    (*iidxs)[++(*isize)] = npts;

    // Reallocate the buffers
    (*iidxs) = REALLOC(*iidxs, unsigned, *isize + 1);
    (*unique) = REALLOC(*unique, unsigned, *isize);

    return 0;
}

/*----------------------------------------------------------------------------*/
/*---------------------- Intensity scaling criterions ------------------------*/
/*----------------------------------------------------------------------------*/
int poisson_likelihood(double *out, double *grad, float *x, unsigned *ij, size_t *dims, unsigned *I0, float *bgd,
                       float *xtal_bi, float *rp, unsigned *fidxs, size_t fsize, unsigned *idxs, size_t isize,
                       unsigned *hkl_idxs, size_t hkl_size, unsigned *oidxs, size_t osize, unsigned threads)
{
    /* Check parameters */
    if (!out || !grad || !x || !ij || !I0 || !bgd || !xtal_bi || !rp || !fidxs || !idxs || !hkl_idxs || !oidxs)
    {ERROR("poisson_likelihood: one of the arguments is NULL."); return -1;}

    if (fsize == 0 || isize == 0 || osize == 0) return 0;

    #pragma omp parallel num_threads(threads)
    {
        int i, j;
        double y_hat, I0_hat;

        double *img = calloc(dims[0] * dims[1], sizeof(double));
        double *obuf = calloc(osize, sizeof(double));
        double *gbuf = calloc(hkl_size, sizeof(double));

        #pragma omp for
        for (i = 0; i < (int)fsize; i++)
        {
            for (j = fidxs[i]; j < (int)fidxs[i + 1]; j++)
            {
                img[ij[j]] += exp(x[isize + hkl_idxs[idxs[j]]]) * xtal_bi[j] * rp[j] + x[idxs[j]];
            }

            for (j = fidxs[i]; j < (int)fidxs[i + 1]; j++)
            {
                I0_hat = img[ij[j]] + bgd[j];
                if (I0_hat > 0.0f)
                {
                    y_hat = exp(x[isize + hkl_idxs[idxs[j]]]) * xtal_bi[j] * rp[j];
                    obuf[oidxs[idxs[j]]] -= I0[j] * log(I0_hat) - I0_hat;
                    grad[idxs[j]] -= I0[j] / I0_hat - 1.0;
                    gbuf[hkl_idxs[idxs[j]]] -= I0[j] / I0_hat * y_hat - y_hat;
                }
            }

            memset(img, 0, dims[0] * dims[1] * sizeof(double));
        }

        for (i = 0; i < (int)osize; i++)
        {
            #pragma omp atomic
            out[i] += obuf[i];
        }
        for (i = 0; i < (int)hkl_size; i++)
        {
            #pragma omp atomic
            grad[isize + i] += gbuf[i];
        }

        DEALLOC(img); DEALLOC(gbuf); DEALLOC(obuf);
    }

    return 0;
}

int least_squares(double *out, double *grad, float *x, unsigned *ij, size_t *dims, unsigned *I0, float *bgd, float *xtal_bi,
                  float *rp, unsigned *fidxs, size_t fsize, unsigned *idxs, size_t isize, unsigned *hkl_idxs, size_t hkl_size,
                  unsigned *oidxs, size_t osize, float (*loss_func)(float), float (*grad_func)(float), unsigned threads)
{
    /* Check parameters */
    if (!out || !grad || !x || !ij || !I0 || !bgd || !xtal_bi || !rp || !fidxs || !idxs || !hkl_idxs || !oidxs)
    {ERROR("least_squares: one of the arguments is NULL."); return -1;}

    if (fsize == 0 || isize == 0 || osize == 0) return 0;

    #pragma omp parallel num_threads(threads)
    {
        int i, j;
        double std, y_hat, I0_hat;

        double *img = calloc(dims[0] * dims[1], sizeof(double));
        double *obuf = calloc(osize, sizeof(double));
        double *gbuf = calloc(hkl_size, sizeof(double));

        #pragma omp for
        for (i = 0; i < (int)fsize; i++)
        {
            for (j = fidxs[i]; j < (int)fidxs[i + 1]; j++)
            {
                img[ij[j]] += exp(x[isize + hkl_idxs[idxs[j]]]) * xtal_bi[j] * rp[j] + x[idxs[j]];
            }

            for (j = fidxs[i]; j < (int)fidxs[i + 1]; j++)
            {
                I0_hat = img[ij[j]] + bgd[j];
                if (I0_hat > 0.0f)
                {
                    std = sqrt(I0_hat); y_hat = exp(x[isize + hkl_idxs[idxs[j]]]) * xtal_bi[j] * rp[j];
                    obuf[oidxs[idxs[j]]] += loss_func((I0[j] - I0_hat) / std);
                    grad[idxs[j]] -= grad_func((I0[j] - I0_hat) / std) / std;
                    gbuf[hkl_idxs[idxs[j]]] -= grad_func((I0[j] - I0_hat) / std) * y_hat / std;
                }
            }

            memset(img, 0, dims[0] * dims[1] * sizeof(double));
        }

        for (i = 0; i < (int)osize; i++)
        {
            #pragma omp atomic
            out[i] += obuf[i];
        }
        for (i = 0; i < (int)hkl_size; i++)
        {
            #pragma omp atomic
            grad[isize + i] += gbuf[i];
        }

        DEALLOC(img); DEALLOC(gbuf); DEALLOC(obuf);
    }

    return 0;
}

int unmerge_sgn(float *I_hat, float *x, unsigned *ij, size_t *dims, unsigned *I0, float *bgd, float *xtal_bi, float *rp,
                unsigned *fidxs, size_t fsize, unsigned *idxs, size_t isize, unsigned *hkl_idxs, size_t hkl_size,
                unsigned threads)
{
    if (!I_hat || !x || !ij || !bgd || !xtal_bi || !rp || !fidxs || !idxs || !hkl_idxs)
    {ERROR("unmerge_sgn: one of the arguments is NULL."); return -1;}

    if (fsize == 0 || isize == 0) return 0;

    #pragma omp parallel num_threads(threads)
    {
        int i, j;
        float ratio;
        
        float *img = calloc(dims[0] * dims[1], sizeof(float));
        float *intercept = calloc(dims[0] * dims[1], sizeof(float));

        #pragma omp for
        for (i = 0; i < (int)fsize; i++)
        {
            for (j = fidxs[i]; j < (int)fidxs[i + 1]; j++)
            {
                img[ij[j]] += expf(x[isize + hkl_idxs[idxs[j]]]) * xtal_bi[j] * rp[j];
                intercept[ij[j]] += x[idxs[j]];
            }

            for (j = fidxs[i]; j < (int)fidxs[i + 1]; j++)
            {
                ratio = (expf(x[isize + hkl_idxs[idxs[j]]]) * xtal_bi[j] * rp[j]) / img[ij[j]];
                CLIP(ratio, 0.0f, 1.0f);
                I_hat[j] = (I0[j] - bgd[j] - intercept[ij[j]]) * ratio;
            }

            memset(img, 0, dims[0] * dims[1] * sizeof(float));
            memset(intercept, 0, dims[0] * dims[1] * sizeof(float));
        }
    }

    return 0;
}