#include "fft_functions.h"
#include "pocket_fft.h"

typedef int (*fft_func)(void *plan, double complex *inp);
typedef int (*rfft_func)(void *plan, double *inp, size_t npts);

static int fft_convolve_calc(void *rfft_plan, void *irfft_plan, line out, double *inp,
    double *krn, size_t flen, rfft_func rfft, rfft_func irfft)
{
    int fail = 0;
    fail = rfft(rfft_plan, inp, flen);
    double re, im;
    for (int i = 0; i < (int)flen / 2 + 1; i++)
    {
        re = (inp[2 * i] * krn[2 * i] - inp[2 * i + 1] * krn[2 * i + 1]);
        im = (inp[2 * i] * krn[2 * i + 1] + inp[2 * i + 1] * krn[2 * i]);
        inp[2 * i] = re; inp[2 * i + 1] = im;
    }
    fail = irfft(irfft_plan, inp, flen);
    for (int i = 0; i < (int)out->npts / 2; i++) ((double *)out->data)[i * out->stride] = inp[i + flen - out->npts / 2] / flen;
    for (int i = 0; i < (int)out->npts / 2 + (int)out->npts % 2; i++) ((double *)out->data)[(i + out->npts / 2) * out->stride] = inp[i] / flen;
    return fail;
}

int fft_convolve_np(double *out, double *inp, int ndim, size_t *dims,
    double *krn, size_t ksize, int axis, EXTEND_MODE mode, double cval,
    unsigned threads)
{
    /* check parameters */
    if (!out || !inp || !dims || !krn) {ERROR("fft_convole_np: one of the arguments is NULL."); return -1;}
    if (ndim <= 0 || ksize == 0) {ERROR("fft_convolve_np: ndim and ksize must be positive."); return -1;}
    if (axis < 0 || axis >= ndim) {ERROR("fft_convolve_np: invalid axis."); return -1;}
    if (threads == 0) {ERROR("fft_convolve_np: threads must be positive."); return -1;}

    double zerro = 0.;
    array oarr = new_array(ndim, dims, sizeof(double), (void *)out);
    array iarr = new_array(ndim, dims, sizeof(double), (void *)inp);
    line kline = new_line(ksize, 1, sizeof(double), krn);
    
    int fail = 0;
    size_t flen = good_size(iarr->dims[axis] + ksize - 1);
    size_t repeats = iarr->size / iarr->dims[axis];
    threads = (threads > (unsigned) repeats) ? (unsigned) repeats : threads;

    #pragma omp parallel num_threads(threads) reduction(|:fail)
    {
        double *inpft = (double *)malloc(2 * (flen / 2 + 1) * sizeof(double));
        double *krnft = (double *)malloc(2 * (flen / 2 + 1) * sizeof(double));
        rfft_plan plan = make_rfft_plan(flen);

        extend_line((void *)krnft, flen, kline, EXTEND_CONSTANT, (void *)&zerro);
        fail |= rfft_np((void *)plan, krnft, flen);

        line iline = init_line(iarr, axis);
        line oline = init_line(oarr, axis);
        #pragma omp for
        for (int i = 0; i < (int)repeats; i++)
        {
            update_line(iline, iarr, i);
            update_line(oline, oarr, i);
            extend_line((void *)inpft, flen, iline, mode, (void *)&cval);
            fail |= fft_convolve_calc((void *)plan, (void *)plan, oline, inpft, krnft,
                flen, rfft_np, irfft_np);
        }

        free(iline); free(oline);
        destroy_rfft_plan(plan);
        free(inpft); free(krnft);    
    }

    free_array(iarr);
    free_array(oarr);
    free(kline);

    return fail;
}

int fft_convolve_fftw(double *out, double *inp, int ndim, size_t *dims,
    double *krn, size_t ksize, int axis, EXTEND_MODE mode, double cval,
    unsigned threads)
{
    /* check parameters */
    if (!out || !inp || !dims || !krn) {ERROR("fft_convolve_np: one of the arguments is NULL."); return -1;}
    if (ndim <= 0 || ksize == 0) {ERROR("fft_convolve_np: ndim and ksize must be positive."); return -1;}
    if (axis < 0 || axis >= ndim) {ERROR("fft_convolve_np: invalid axis."); return -1;}
    if (threads == 0) {ERROR("fft_convolve_np: threads must be positive."); return -1;}

    double zerro = 0.;
    array oarr = new_array(ndim, dims, sizeof(double), (void *)out);
    array iarr = new_array(ndim, dims, sizeof(double), (void *)inp);
    line kline = new_line(ksize, 1, sizeof(double), krn);
    
    int fail = 0;
    size_t flen = next_fast_len_fftw(iarr->dims[axis] + ksize - 1);
    size_t repeats = iarr->size / iarr->dims[axis];
    threads = (threads > (unsigned)repeats) ? (unsigned)repeats : threads;

    #pragma omp parallel num_threads(threads) reduction(|:fail)
    {
        double *inpft = (double *)fftw_malloc(2 * (flen / 2 + 1) * sizeof(double));
        double *krnft = (double *)fftw_malloc(2 * (flen / 2 + 1) * sizeof(double));
        fftw_iodim *dim = (fftw_iodim *)malloc(sizeof(fftw_iodim));
        dim->n = flen; dim->is = 1; dim->os = 1;
        fftw_plan rfft_plan, irfft_plan;

        #pragma omp critical
        {
            rfft_plan = fftw_plan_guru_dft_r2c(1, dim, 0, NULL, inpft, (fftw_complex *)inpft,
                FFTW_ESTIMATE);
            irfft_plan = fftw_plan_guru_dft_c2r(1, dim, 0, NULL, (fftw_complex *)inpft,
                inpft, FFTW_ESTIMATE);
        }

        extend_line((void *)krnft, flen, kline, EXTEND_CONSTANT, (void *)&zerro);
        fail |= rfft_fftw((void *)rfft_plan, krnft, flen);

        line iline = init_line(iarr, axis);
        line oline = init_line(oarr, axis);
        #pragma omp for
        for (int i = 0; i < (int)repeats; i++)
        {
            update_line(iline, iarr, i);
            update_line(oline, oarr, i);
            extend_line((void *)inpft, flen, iline, mode, (void *)&cval);
            fail |= fft_convolve_calc((void *)rfft_plan, (void *)irfft_plan, oline,
                inpft, krnft, flen, rfft_fftw, irfft_fftw);
        }

        free(iline); free(oline);
        fftw_destroy_plan(rfft_plan);
        fftw_destroy_plan(irfft_plan);
        free(dim); fftw_free(inpft); fftw_free(krnft);
    }

    free_array(iarr);
    free_array(oarr);
    free(kline);

    return fail;
}

int gauss_kernel1d(double *out, double sigma, unsigned order, size_t ksize)
{
    /* check parameters */
    if (!out) {ERROR("gauss_kernel1d: out is NULL."); return -1;}
    if (sigma <= 0) {ERROR("gauss_kernel1d: sigma must be positive."); return -1;}
    if (!ksize) {ERROR("gauss_kernel1d: ksize must be positive."); return -1;}

    int radius = (ksize - 1) / 2;
    double sum = 0;
    double sigma2 = sigma * sigma;
    for (int i = 0; i < (int) ksize; i++)
    {
        out[i] = exp(-0.5 * pow(i - radius, 2) / sigma2); sum += out[i];
    }
    for (int i = 0; i < (int) ksize; i++) out[i] /= sum;
    if (order)
    {
        double *q0 = (double *)calloc(order + 1, sizeof(double)); q0[0] = 1.;
        double *q1 = (double *)calloc(order + 1, sizeof(double));
        int idx; double qval;
        for (int k = 0; k < (int) order; k++)
        {
            for (int i = 0; i <= (int) order; i++)
            {
                qval = 0;
                for (int j = 0; j <= (int) order; j++)
                {
                    idx = j + (order + 1) * i;
                    if ((idx % (order + 2)) == 1) qval += q0[j] * (idx / (order + 2) + 1);
                    if ((idx % (order + 2)) == (order + 1)) qval -= q0[j] / sigma2; 
                }
                q1[i] = qval;
            }
            for (int i = 0; i <= (int) order; i++) q0[i] = q1[i];
        }
        free(q0);
        double fct;
        for (int i = 0; i < (int) ksize; i++)
        {
            fct = 0;
            for (int j = 0; j <= (int) order; j++) fct += pow(i - radius, j) * q1[j];
            out[i] *= fct;
        }
        free(q1);
    }
    return 0;
}

int gauss_filter(double *out, double *inp, int ndim, size_t *dims, double *sigma,
    unsigned *order, EXTEND_MODE mode, double cval, double truncate, unsigned threads,
    convolve_func fft_convolve)
{
    /* check parameters */
    if (!out || !inp || !dims || !sigma || !order)
    {ERROR("gauss_filter: one of the arguments is NULL."); return -1;}
    if (ndim <= 0) {ERROR("gauss_filter: ndim must be positive."); return -1;}
    if (!threads) {ERROR("gauss_filter: threads must be positive."); return -1;}

    int fail = 0;
    int axis = 0;
    while (sigma[axis] < 1e-15 && axis < ndim) axis++;
    if (axis < ndim)
    {
        size_t ksize = 2 * (size_t) (sigma[axis] * truncate) + 1;
        double *krn = (double *)malloc(ksize * sizeof(double));
        fail |= gauss_kernel1d(krn, sigma[axis], order[axis], ksize);
        fail |= fft_convolve(out, inp, ndim, dims, krn, ksize, axis, mode, cval, threads);
        free(krn);

        for (int n = axis + 1; n < ndim; n++)
        {
            if (sigma[n] > 1e-15)
            {
                ksize = 2 * (size_t) (sigma[n] * truncate) + 1;
                krn = (double *)malloc(ksize * sizeof(double));
                fail |= gauss_kernel1d(krn, sigma[n], order[n], ksize);
                fail |= fft_convolve(out, out, ndim, dims, krn, ksize, n, mode, cval, threads);
                free(krn);
            }
        }
    }
    else
    {
        size_t size = 1;
        for (int n = 0; n < ndim; n++) size *= dims[n];
        #pragma omp parallel for num_threads(threads)
        for (int i = 0; i < (int)size; i++) out[i] = inp[i];
    }
    return fail;
}

int gauss_grad_mag(double *out, double *inp, int ndim, size_t *dims, double *sigma,
    EXTEND_MODE mode, double cval, double truncate, unsigned threads,
    convolve_func fft_convolve)
{
    /* check parameters */
    if (!out || !inp || !dims || !sigma)
    {ERROR("gauss_grad_mag: one of the arguments is NULL."); return -1;}
    if (ndim <= 0) {ERROR("gauss_grad_mag: ndim must be positive."); return -1;}
    if (!threads) {ERROR("gauss_grad_mag: threads must be positive."); return -1;}

    int fail = 0;
    size_t size = 1;
    unsigned *order = (unsigned *)malloc(ndim * sizeof(unsigned));
    for (int n = 0; n < ndim; n++) {order[n] = (n == 0) ? 1 : 0; size *= dims[n];}

    fail |= gauss_filter(out, inp, ndim, dims, sigma, order, mode,
        cval, truncate, threads, fft_convolve);

    #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < (int)size; i++) out[i] = out[i] * out[i];

    double *tmp = (double *)malloc(size * sizeof(double));
    for (int m = 1; m < ndim; m++)
    {
        for (int n = 0; n < ndim; n++) order[n] = (n == m) ? 1 : 0;
        fail |= gauss_filter(tmp, inp, ndim, dims, sigma, order, mode,
            cval, truncate, threads, fft_convolve);
        
        #pragma omp parallel for num_threads(threads)
        for (int i = 0; i < (int)size; i++) out[i] += tmp[i] * tmp[i];
    }

    free(tmp); free(order);
    #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < (int)size; i++) out[i] = sqrt(out[i]);
    return fail;
}