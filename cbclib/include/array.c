#include "array.h"

array new_array(int ndim, size_t *dims, size_t item_size, void *data)
{
    /* check parameters */
    if(ndim <= 0) {ERROR("new_array: ndim must be positive."); return NULL;}

    array arr = (array)malloc(sizeof(struct array_s));
    if (!arr) {ERROR("new_array: not enough memory."); return NULL;}

    arr->ndim = ndim;
    arr->item_size = item_size;
    arr->size = 1;
    for (int n = 0; n < ndim; n++) arr->size *= dims[n];

    arr->dims = dims;
    arr->strides = (size_t *)malloc(arr->ndim * sizeof(size_t));
    if (!arr->strides) {ERROR("new_array: not enough memory."); return NULL;}
    size_t stride = 1;
    for (int n = arr->ndim - 1; n >= 0; n--)
    {
        arr->strides[n] = stride;
        stride *= arr->dims[n];
    }
    arr->data = data;
    return arr;
}

void free_array(array arr)
{
    free(arr->strides);
    free(arr);
}

// note: the line count over axis is given by: arr->size / arr->dims[axis]
// note: you can free the line just with: free(line)

line new_line(size_t npts, size_t stride, size_t item_size, void *data)
{
    line ln = (line)malloc(sizeof(line_s));
    if (!ln) {ERROR("new_line: not enough memory."); return NULL;}

    ln->npts = npts;
    ln->stride = stride;
    ln->item_size = item_size;
    ln->line_size = ln->npts * ln->stride * ln->item_size;
    ln->data = data;
    return ln;
}

line init_line(array arr, int axis)
{
    /* check parameters */
    if (axis < 0 || axis >= arr->ndim) {ERROR("init_line: invalid axis."); return NULL;}

    line ln = (line)malloc(sizeof(line_s));
    if (!ln) {ERROR("init_line: not enough memory."); return NULL;}

    ln->npts = arr->dims[axis];
    ln->stride = arr->strides[axis];
    ln->item_size = arr->item_size;
    ln->line_size = ln->npts * ln->stride * ln->item_size;
    ln->data = arr->data;
    return ln;
}

void extend_line(void *out, size_t osize, line inp, EXTEND_MODE mode, void *cval)
{
    int dsize = (int)osize - (int)inp->npts;
    int size_before = dsize - dsize / 2;
    int size_after = dsize - size_before;

    void *last = inp->data + inp->line_size;
    void *dst = out + size_before * inp->item_size;
    void *src = inp->data;

    int line_size = inp->npts;
    while(line_size--)
    {
        memcpy(dst, src, inp->item_size);
        dst += inp->item_size;
        src += inp->stride * inp->item_size;
    }

    switch (mode)
    {
        /* kkkkkkkk|abcd|kkkkkkkk */
        case EXTEND_CONSTANT:

            dst = out;
            while (size_before--)
            {
                memcpy(dst, cval, inp->item_size);
                dst += inp->item_size;
            }

            dst = out + (osize - size_after) * inp->item_size;
            while (size_after--)
            {
                memcpy(dst, cval, inp->item_size);
                dst += inp->item_size;
            }
            break;

        /* aaaaaaaa|abcd|dddddddd */
        case EXTEND_NEAREST:

            dst = out; src = inp->data;
            while (size_before--)
            {
                memcpy(dst, src, inp->item_size);
                dst += inp->item_size;
            }

            dst = out + (osize - size_after) * inp->item_size;
            src = last - inp->stride * inp->item_size;
            while (size_after--)
            {
                memcpy(dst, src, inp->item_size);
                dst += inp->item_size;
            }
            break;

        /* cbabcdcb|abcd|cbabcdcb */
        case EXTEND_MIRROR:

            dst = out + (size_before - 1) * inp->item_size;
            src = inp->data + inp->stride * inp->item_size;

            while (size_before-- && src < last)
            {
                memcpy(dst, src, inp->item_size);
                dst -= inp->item_size;
                src += inp->item_size * inp->stride;
            }
            src = last - 2 * inp->stride * inp->item_size;
            while (size_before-- >= 0 && src >= inp->data)
            {
                memcpy(dst, src, inp->item_size);
                dst -= inp->item_size;
                src -= inp->item_size * inp->stride;
            }

            dst = out + (osize - size_after) * inp->item_size;
            src = last - 2 * inp->stride * inp->item_size;

            while (size_after-- && src >= inp->data)
            {
                memcpy(dst, src, inp->item_size);
                dst += inp->item_size;
                src -= inp->item_size * inp->stride;
            }
            src = inp->data + inp->stride * inp->item_size;
            while (size_after-- >= 0 && src < last)
            {
                memcpy(dst, src, inp->item_size);
                dst += inp->item_size;
                src += inp->item_size * inp->stride;
            }
            break;

        /* abcddcba|abcd|dcbaabcd */
        case EXTEND_REFLECT:
            dst = out + (size_before - 1) * inp->item_size;
            src = inp->data;

            while (size_before-- && src < last)
            {
                memcpy(dst, src, inp->item_size);
                dst -= inp->item_size;
                src += inp->item_size * inp->stride;
            }
            src = last - inp->stride * inp->item_size;
            while (size_before-- >= 0 && src >= inp->data)
            {
                memcpy(dst, src, inp->item_size);
                dst -= inp->item_size;
                src -= inp->item_size * inp->stride;
            }

            dst = out + (osize - size_after) * inp->item_size;
            src = last - inp->stride * inp->item_size;

            while (size_after-- && src >= inp->data)
            {
                memcpy(dst, src, inp->item_size);
                dst += inp->item_size;
                src -= inp->item_size * inp->stride;
            }
            src = inp->data;
            while (size_after-- >= 0 && src < last)
            {
                memcpy(dst, src, inp->item_size);
                dst += inp->item_size;
                src += inp->item_size * inp->stride;
            }
            break;

        /* abcdabcd|abcd|abcdabcd */
        case EXTEND_WRAP:
            dst = out + (size_before - 1) * inp->item_size;
            src = last - inp->stride * inp->item_size;

            while (size_before-- && src >= inp->data)
            {
                memcpy(dst, src, inp->item_size);
                dst -= inp->item_size;
                src -= inp->item_size * inp->stride;
            }

            src = last - inp->stride * inp->item_size;
            while (size_before-- >= 0 && src >= inp->data)
            {
                memcpy(dst, src, inp->item_size);
                dst -= inp->item_size;
                src -= inp->item_size * inp->stride;
            }

            dst = out + (osize - size_after) * inp->item_size;
            src = inp->data;

            while (size_after-- && src < last)
            {
                memcpy(dst, src, inp->item_size);
                dst += inp->item_size;
                src += inp->item_size * inp->stride;
            }
            src = inp->data;
            while (size_after-- >= 0 && src < last)
            {
                memcpy(dst, src, inp->item_size);
                dst += inp->item_size;
                src += inp->item_size * inp->stride;
            }
            break;

        default:
            ERROR("extend_line: invalid extend mode.");
    }
}

int extend_point(void *out, int *coord, array arr, array mask, EXTEND_MODE mode, void *cval)
{
    /* kkkkkkkk|abcd|kkkkkkkk */
    if (mode == EXTEND_CONSTANT)
    {
            memcpy(out, cval, arr->item_size);
            return 1;
    }

    int *close = (int *)malloc(arr->ndim * sizeof(int));
    size_t dist;

    switch (mode)
    {
        /* aaaaaaaa|abcd|dddddddd */
        case EXTEND_NEAREST:

            for (int n = 0; n < arr->ndim; n++)
            {
                if (coord[n] >= (int)arr->dims[n]) close[n] = arr->dims[n] - 1;
                else if (coord[n] < 0) close[n] = 0;
                else close[n] = coord[n];
            }

            break;

        /* cbabcdcb|abcd|cbabcdcb */
        case EXTEND_MIRROR:

            for (int n = 0; n < arr->ndim; n++)
            {
                if (coord[n] >= (int)arr->dims[n])
                {
                    close[n] = arr->dims[n] - 1;
                    dist = coord[n] - arr->dims[n] + 1;

                    while(dist-- && close[n] >= 0) close[n]--;
                }
                else if (coord[n] < 0)
                {
                    close[n] = 0; dist = -coord[n];

                    while(dist-- && close[n] < (int)arr->dims[n]) close[n]++;
                }
                else close[n] = coord[n];
            }

            break;

        /* abcddcba|abcd|dcbaabcd */
        case EXTEND_REFLECT:

            for (int n = 0; n < arr->ndim; n++)
            {
                if (coord[n] >= (int)arr->dims[n])
                {
                    close[n] = arr->dims[n] - 1;
                    dist = coord[n] - arr->dims[n];

                    while(dist-- && close[n] >= 0) close[n]--;
                }
                else if (coord[n] < 0)
                {
                    close[n] = 0; dist = -coord[n] - 1;

                    while(dist-- && close[n] < (int)arr->dims[n]) close[n]++;
                }
                else close[n] = coord[n];
            }

            break;

        /* abcdabcd|abcd|abcdabcd */
        case EXTEND_WRAP:

            for (int n = 0; n < arr->ndim; n++)
            {
                if (coord[n] >= (int)arr->dims[n])
                {
                    close[n] = 0;
                    dist = coord[n] - arr->dims[n];

                    while(dist-- && close[n] < (int)arr->dims[n]) close[n]++;
                }
                else if (coord[n] < 0)
                {
                    close[n] = arr->dims[n] - 1;
                    dist = -coord[n] - 1;

                    while(dist-- && close[n] >= 0) close[n]--;
                }
                else close[n] = coord[n];
            }

            break;

        default:
            ERROR("extend_point: invalid extend mode.");
    }

    int index;
    RAVEL_INDEX(close, &index, arr);
    free(close);

    if (((unsigned char *)mask->data)[index])
    {
        memcpy(out, arr->data + index * arr->item_size, arr->item_size);
        return 1;
    }
    else return 0;

}

static size_t binary_search(const void *key, const void *array, size_t l, size_t r, size_t size,
    int (*compar)(const void*, const void*))
{
    if (l <= r)
    {
        size_t m = l + (r - l) / 2;
        int cmp0 = compar(key, array + m * size);
        int cmp1 = compar(key, array + (m + 1) * size);
        if (cmp0 == 0) return m;
        if (cmp0 > 0 && cmp1 < 0) return m + 1;
        if (cmp0 < 0) return binary_search(key, array, l, m, size, compar);
        return binary_search(key, array, m + 1, r, size, compar);
    }
    return 0;
}

size_t searchsorted(const void *key, const void *base, size_t npts, size_t size,
    int (*compar)(const void*, const void*))
{
    if (compar(key, base) < 0) return 0;
    if (compar(key, base + (npts - 1) * size) > 0) return npts;
    return binary_search(key, base, 0, npts, size, compar);
}

static void setPixelColor(array image, int x, int y, unsigned int val, unsigned int max_val)
{
    unsigned int *ptr = image->data + (image->strides[1] * x + image->strides[0] * y) * image->item_size;
    unsigned int new = *ptr + val;
    *ptr = new > max_val ? max_val : new;
}

static int confine(int coord, int axis, array arr)
{
    coord = coord > 0 ? coord : 0;
    coord = coord < (int)arr->dims[axis] ? coord : (int)arr->dims[axis] - 1;
    return coord;
}

static void plotLineWidth(array image, int x0, int y0, int x1, int y1, double wd, unsigned int max_val)
{
    /* plot an anti-aliased line of width wd */
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx - dy, derr = 0, dx0 = 0, e2, x2, y2, val;    /* error value e_xy */
    double ed = dx + dy == 0 ? 1 : sqrt((double)dx * dx + (double)dy * dy);
    wd = (wd + 1) / 2;

    /* define line bounds */
    int wi = wd, x_min, x_max, y_min, y_max;
    if (x0 < x1)
    {
        x_min = confine(x0 - wi, 1, image);
        x_max = confine(x1 + wi, 1, image);
        err += (x0 - x_min) * dy; x0 = x_min; x1 = x_max;
    }
    else
    {
        x_min = confine(x1 - wi, 1, image);
        x_max = confine(x0 + wi, 1, image);
        err += (x_max - x0) * dy; x0 = x_max; x1 = x_min;
    }
    if (y0 < y1)
    {
        y_min = confine(y0 - wi, 0, image);
        y_max = confine(y1 + wi, 0, image);
        err -= (y0 - y_min) * dx; y0 = y_min; y1 = y_max;
    }
    else
    {
        y_min = confine(y1 - wi, 0, image);
        y_max = confine(y0 + wi, 0, image);
        err -= (y_max - y0) * dx; y0 = y_max; y1 = y_min;
    }

    while (1)
    {
        /* pixel loop */
        err += derr; derr = 0;
        x0 += dx0; dx0 = 0;
        val = max_val - fmax(max_val * (abs(err - dx + dy) / ed - wd + 1), 0.0);
        setPixelColor(image, x0, y0, val, max_val);

        if (2 * err >= -dx)
        {
            /* x step */
            for (e2 = err + dy, y2 = y0 + sy; abs(e2) < ed * wd && y2 >= y_min && y2 <= y_max; e2 += dx, y2 += sy)
            {
                val = max_val - fmax(max_val * (abs(e2) / ed - wd + 1), 0.0);
                setPixelColor(image, x0, y2, val, max_val);
            }
            if (x0 == x1) break;
            derr -= dy; dx0 += sx;
        }
        if (2 * err <= dy)
        {
            /* y step */
            for (e2 = err - dx, x2 = x0 + sx; abs(e2) < ed * wd && x2 >= x_min && x2 <= x_max; e2 -= dy, x2 += sx)
            {
                val = max_val - fmax(max_val * (abs(e2) / ed - wd + 1), 0.0);
                setPixelColor(image, x2, y0, val, max_val);
            }
            if (y0 == y1) break;
            derr += dx; y0 += sy;
        }
    }
}

int draw_lines(unsigned int *out, size_t X, size_t Y, unsigned int max_val, double *lines, size_t n_lines, unsigned int dilation)
{
    /* check parameters */
    if (!out || !lines) {ERROR("line_draw: one of the arguments is NULL."); return -1;}
    if (X == 0 || Y == 0) {ERROR("line_draw: image size must be positive."); return -1;}

    size_t ldims[2] = {n_lines, 7};
    size_t odims[2] = {Y, X};
    array larr = new_array(2, ldims, sizeof(double), lines);
    array oarr = new_array(2, odims, sizeof(unsigned int), out);

    line ln = init_line(larr, 1);
    int x0, y0, x1, y1;
    double wd;

    for (int i = 0; i < (int)n_lines; i++)
    {
        UPDATE_LINE(ln, larr, i);

        x0 = *(double *)ln->data;
        y0 = *(double *)(ln->data + ln->item_size);
        x1 = *(double *)(ln->data + 2 * ln->item_size);
        y1 = *(double *)(ln->data + 3 * ln->item_size);
        wd = *(double *)(ln->data + 4 * ln->item_size);

        plotLineWidth(oarr, x0, y0, x1, y1, wd + (double)dilation, max_val);
    }

    free(ln);
    free_array(larr); free_array(oarr);

    return 0;
}