#include "median.h"

footprint init_footprint(int ndim, size_t item_size, size_t *fsize)
{
    /* check parameters */
    if(ndim <= 0) {ERROR("new_footprint: ndim must be positive."); return NULL;}

    array farr = new_array(ndim, fsize, 0, NULL);

    footprint fpt = (footprint)malloc(sizeof(struct footprint_s));
    fpt->ndim = farr->ndim;
    fpt->npts = farr->size;

    fpt->offsets = (size_t *)malloc(fpt->npts * fpt->ndim * sizeof(size_t));
    fpt->coordinates = (size_t *)malloc(fpt->npts * fpt->ndim * sizeof(size_t));

    fpt->item_size = item_size;
    fpt->data = malloc(fpt->npts * fpt->item_size);
    
    if (!fpt || !fpt->offsets || !fpt->coordinates || !fpt->data)
    {ERROR("new_footprint: not enough memory."); return NULL;}

    for (int i = 0; i < (int)fpt->npts; i++)
    {
        unravel_index(&(fpt->offsets[ndim * i]), i, farr);
    }

    free_array(farr);
    return fpt;
}

void free_footprint(footprint fpt)
{
    free(fpt->coordinates);
    free(fpt->offsets);
    free(fpt->data);
    free(fpt);
}

void update_footprint(footprint fpt, size_t *coord, array arr, EXTEND_MODE mode, void *cval)
{
    int extend, index;

    for (int i = 0; i < fpt->npts; i++)
    {
        extend = 0;

        for (int n = 0; n < fpt->ndim; n++)
        {
            fpt->coordinates[i * fpt->ndim + n] = coord[n] + fpt->offsets[i * fpt->ndim + n];
            extend |= (fpt->coordinates[i * fpt->ndim + n] >= arr->dims[n]) ||
                (fpt->coordinates[i * fpt->ndim + n] < 0);
        }

        if (extend)
        {
            extend_point(fpt->data + i * fpt->item_size, &(fpt->coordinates[i * fpt->ndim]),
                arr, mode, cval);
        }
        else
        {
            index = ravel_index(&(fpt->coordinates[i * fpt->ndim]), arr);
            memcpy(fpt->data + i * fpt->item_size, arr->data + index * arr->item_size,
                arr->item_size);
        }
    }
}

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

static void wirthselect(void *data, void *key, int k, int l, int m, size_t size,
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
    
    key = data + k * size;
}

int median(void *out, void *data, unsigned char *mask, int ndim, size_t *dims, size_t item_size, int axis,
    int (*compar)(const void*, const void*), unsigned threads)
{
    /* check parameters */
    if (!out || !data || !mask || !dims) {ERROR("median: one of the arguments is NULL."); return -1;}
    if (ndim <= 0) {ERROR("median: ndim must be positive."); return -1;}
    if (axis < 0 || axis >= ndim) {ERROR("median: invalid axis."); return -1;}
    if (threads == 0) {ERROR("median: threads must be positive."); return -1;}

    array iarr = new_array(ndim, dims, item_size, data);
    array marr = new_array(ndim, dims, 1, mask);

    int repeats = iarr->size / iarr->dims[axis];
    threads = (threads > (unsigned)repeats) ? (unsigned)repeats : threads;

    #pragma omp parallel num_threads(threads)
    {
        void *buffer = malloc(iarr->dims[axis] * iarr->item_size);
        void *key = malloc(iarr->item_size);

        line iline = init_line(iarr, axis);
        line mline = init_line(marr, axis);

        #pragma omp for
        for (int i = 0; i < (int)repeats; i++)
        {
            update_line(iline, iarr, i);
            update_line(mline, marr, i);

            int len = 0;
            for (int n = 0; n < (int)iline->npts; n++)
            {
                if (((unsigned char *)mline->data)[n * mline->stride])
                {memcpy(buffer + len++ * iline->item_size,
                    iline->data + n * iline->stride * iline->item_size, iline->item_size);}
            }

            if (len) 
            {
                wirthselect(buffer, key, len / 2, 0, len - 1, iline->item_size, compar);
                memcpy(out + i * iline->item_size, key, iline->item_size);
            }
            else memset(out + i * iline->item_size, 0, iline->item_size);

        }

        free(iline); free(mline);
        free(key); free(buffer);
    }

    free_array(iarr); free_array(marr);

    return 0;
}

// int median_filter(void *out, void *data, unsigned char *mask, int ndim, size_t *dims, size_t item_size,
//     int axis, size_t window, EXTEND_MODE mode, void *cval, int (*compar)(const void*, const void*),
//     unsigned threads)
// {
//     /* check parameters */
//     if (!out || !data || !mask || !cval) {ERROR("median_filter: one of the arguments is NULL."); return -1;}
//     if (ndim <= 0) {ERROR("median_filter: ndim must be positive."); return -1;}
//     if (axis < 0 || axis >= ndim) {ERROR("median_filter: invalid axis."); return -1;}
//     if (window == 0) {ERROR("median_filter: window must be positive."); return -1;}
//     if (threads == 0) {ERROR("median_filter: threads must be positive."); return -1;}

//     unsigned char mval = 1;

//     array iarr = new_array(ndim, dims, item_size, data);
//     array oarr = new_array(ndim, dims, item_size, out);
//     array marr = new_array(ndim, dims, 1, (void *)mask);

//     int repeats = iarr->size / iarr->dims[axis];
//     threads = (threads > (unsigned)repeats) ? (unsigned)repeats : threads;

//     #pragma omp parallel num_threads(threads)
//     {
//         void *inpbf = malloc((iarr->dims[axis] + window) * iarr->item_size);
//         unsigned char *mbf = (unsigned char *)malloc(marr->dims[axis] + window);
//         void *medbf = malloc(window * iarr->item_size);
//         void *key = malloc(iarr->item_size);

//         line iline = init_line(iarr, axis);
//         line oline = init_line(oarr, axis);
//         line mline = init_line(marr, axis);

//         #pragma omp for
//         for (int i = 0; i < (int)repeats; i++)
//         {
//             update_line(iline, iarr, i);
//             update_line(oline, oarr, i);
//             update_line(mline, marr, i);

//             extend_line(inpbf, iline->npts + window, iline, mode, cval);
//             extend_line((void *)mbf, mline->npts + window, mline, mode, (void *)&mval);

//             for (int j = 0; j < (int)iline->npts; j++)
//             {
//                 int len = 0;
//                 for (int n = 0; n < window; n++)
//                 {
//                     if (mbf[n + j])
//                     {memcpy(medbf + len++ * iline->item_size,
//                         inpbf + (n + j) * iline->item_size, iline->item_size);}
//                 }
                
//                 if (len) 
//                 {
//                     wirthselect(medbf, key, len / 2, 0, len - 1, item_size, compar);
//                     memcpy(oline->data + j * oline->stride * oline->item_size, key,
//                         oline->item_size);
//                 }
//                 else memset(oline->data + j * oline->stride * oline->item_size, 0, oline->item_size);
//             }
//         }

//         free(iline); free(oline); free(mline);
//         free(key); free(medbf); free(mbf); free(inpbf);

//     }

//     free_array(iarr); free_array(oarr); free_array(marr);

//     return 0;
// }

int median_filter(void *out, void *data, int ndim, size_t *dims, size_t item_size, size_t *fsize,
    EXTEND_MODE mode, void *cval, int (*compar)(const void*, const void*), unsigned threads)
{
    /* check parameters */
    if (!out || !data || !fsize || !cval)
    {ERROR("median_filter: one of the arguments is NULL."); return -1;}
    if (ndim <= 0) {ERROR("median_filter: ndim must be positive."); return -1;}
    if (threads == 0) {ERROR("median_filter: threads must be positive."); return -1;}

    array iarr = new_array(ndim, dims, item_size, data);

    #pragma omp parallel num_threads(threads)
    {
        footprint fpt = init_footprint(iarr->ndim, iarr->item_size, fsize);
        size_t *coord = (size_t *)malloc(iarr->ndim * sizeof(size_t));
        void *key = malloc(iarr->item_size);

        #pragma omp for
        for (int i = 0; i < (int)iarr->size; i++)
        {
            unravel_index(coord, i, iarr);

            update_footprint(fpt, coord, iarr, mode, cval);

            wirthselect(fpt->data, key, fpt->npts / 2, 0, fpt->npts - 1, fpt->item_size, compar);
            memcpy(out + i * fpt->item_size, key, fpt->item_size);
        }

        free_footprint(fpt); free(coord); free(key);
    }

    free_array(iarr);

    return 0;
}