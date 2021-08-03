#include "median.h"

// footprint init_footprint(int ndim, size_t *fsize)
// {
//     /* check parameters */
//     if(ndim <= 0) {ERROR("new_footprint: ndim must be positive."); return NULL;}
//     if(npts <= 0) {ERROR("new_footprint: npts must be positive."); return NULL;}

//     footprint fpt = (footprint)malloc(sizeof(struct footprint));
//     arr->offsets = (size_t *)malloc(npts * sizeof(size_t));
//     arr->idxs = (size_t *)malloc(npts * sizeof(size_t));
//     if (!arr || !arr->offsets || !arr->idxs)
//     {ERROR("new_footprint: not enough memory."); return NULL;}

//     int npts = 1;
//     for (int n = 0; n < ndim; n++) npts *= fsize[n];
//     for (int i = 0; i < npts; i++)
//     {
//         for (int n = 0; n < ndim; n++)
//         {offsets[n + i * ndim] = i / fsize[n];}
//     }
// }

int compare_double(const void *a, const void *b)
{
    if (*(double*)a > *(double*)b) return 1;
    else if (*(double*)a < *(double*)b) return -1;
    else return 0;
}

int compare_float(const void *a, const void *b)
{
    if (*(float*)a > *(float*)b) return 1;
    else if (*(float*)a < *(float*)b) return -1;
    else return 0;
}

int compare_long(const void *a, const void *b)
{
    return (*(long *)a - *(long *)b);
}

static void *wirthselect(void *data, void *key, int k, int l, int m, size_t size,
    int (*compar)(const void*, const void*))
{
    int i, j;
    while (l < m)
    {
        memcpy(key, data + k * size, size);
        i = l; j = m;

        do
        {
            while (compar(key, data + i * size) > 0) i++;
            while (compar(key, data + j * size) < 0) j--;
            if (i <= j) 
            {
                SWAP_BUF(data + i * size, data + j * size, size);
                i++; j--;
            }
        } while((i <= j));
        if (j < k) l = i;
        if (k < i) m = j;
    }
    
    return data + k * size;
}

int median(void *out, void *data, unsigned char *mask, int ndim, size_t *dims, size_t item_size, int axis,
    int (*compar)(const void*, const void*), unsigned threads)
{
    /* check parameters */
    if (!out || !data || !mask || !dims) {ERROR("median: one of the arguments is NULL."); return -1;}
    if (ndim <= 0) {ERROR("median: ndim must be positive."); return -1;}
    if (axis < 0 || axis >= ndim) {ERROR("median: invalid axis."); return -1;}
    if (threads == 0) {ERROR("median: threads must be positive."); return -1;}

    array iarr = new_array(ndim, dims, data);
    array marr = new_array(ndim, dims, mask);

    int repeats = iarr->size / iarr->dims[axis];
    threads = (threads > (unsigned)repeats) ? (unsigned)repeats : threads;

    #pragma omp parallel num_threads(threads)
    {
        unsigned char *buffer = (unsigned char *)malloc(iarr->dims[axis] * item_size);
        void *key = malloc(item_size);

        line iline = init_line(iarr, axis);
        line mline = init_line(marr, axis);
        #pragma omp for
        for (int i = 0; i < (int)repeats; i++)
        {
            update_line(iline, iarr, i, item_size);
            update_line(mline, marr, i, 1);

            int len = 0;
            for (int n = 0; n < (int)iline->npts; n++)
            {
                if (((unsigned char *)mline->data)[n * mline->stride])
                {memcpy(buffer + len++ * item_size, iline->data + n * iline->stride * item_size, item_size);}
            }
            if (len) 
            {
                void *median = wirthselect(buffer, key, (len & 1) ? (len / 2) : (len / 2 - 1),
                    0, len - 1, item_size, compar);
                memcpy(out + i * item_size, median, item_size);
            }
            else memset(out + i * item_size, 0, item_size);
        }
        free(key); free(buffer);
    }

    return 0;
}

int median_filter(void *out, void *data, unsigned char *mask, int ndim, size_t *dims, size_t item_size,
    int axis, size_t window, EXTEND_MODE mode, void *cval, int (*compar)(const void*, const void*),
    unsigned threads)
{
    /* check parameters */
    if (!out || !data || !mask || !cval) {ERROR("median_filter: one of the arguments is NULL."); return -1;}
    if (ndim <= 0) {ERROR("median_filter: ndim must be positive."); return -1;}
    if (axis < 0 || axis >= ndim) {ERROR("median_filter: invalid axis."); return -1;}
    if (window == 0) {ERROR("median_filter: window must be positive."); return -1;}
    if (threads == 0) {ERROR("median_filter: threads must be positive."); return -1;}

    unsigned char mval = 1;
    array iarr = new_array(ndim, dims, data);
    array oarr = new_array(ndim, dims, out);
    array marr = new_array(ndim, dims, (void *)mask);

    int repeats = iarr->size / iarr->dims[axis];
    threads = (threads > (unsigned)repeats) ? (unsigned)repeats : threads;

    #pragma omp parallel num_threads(threads)
    {
        void *inpbf = malloc((iarr->dims[axis] + window) * item_size);
        unsigned char *mbf = (unsigned char *)malloc(marr->dims[axis] + window);
        void *medbf = malloc(window * item_size);
        void *key = malloc(item_size);

        line iline = init_line(iarr, axis);
        line oline = init_line(oarr, axis);
        line mline = init_line(marr, axis);

        #pragma omp for
        for (int i = 0; i < (int)repeats; i++)
        {
            update_line(iline, iarr, i, item_size);
            update_line(oline, oarr, i, item_size);
            update_line(mline, marr, i, 1);

            extend_line(inpbf, item_size, iarr->dims[axis] + window, iline, mode, cval);
            extend_line((void *)mbf, 1, marr->dims[axis] + window, mline, mode, (void *)&mval);

            for (int j = 0; j < (int)iline->npts; j++)
            {
                int len = 0;
                for (int n = -(int)window / 2; n < (int)window / 2 + (int)window % 2; n++)
                {
                    if (mbf[n + j])
                    {memcpy(medbf + len++ * item_size, inpbf + (n + j) * item_size, item_size);}
                }
                if (len) 
                {
                    void *median = wirthselect(medbf, key, (len & 1) ? (len / 2) : (len / 2 - 1),
                        0, len - 1, item_size, compar);
                    memcpy(oline->data + j * oline->stride * item_size, median, item_size);
                }
                else memset(oline->data + j * oline->stride * item_size, 0, item_size);
            }
        }
        free(iline); free(oline); free(mline);
        free(key); free(medbf); free(mbf); free(inpbf);
    }

    return 0;
}