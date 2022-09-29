#include "img_proc.h"
#include "array.h"

#define TOL 3.1425926535897937e-05

#define CLIP(c, a, b)                           \
{                                               \
    c = ((c) > (a)) ? (c) : (a);                \
    c = ((c) < (b)) ? (c) : (b);                \
}

#define WRAP_DIST(_dist, _dx, _hb, _fb, _div)   \
{                                               \
    float _dx1;                                 \
    if (_dx < -_hb) _dx1 = _dx + _fb;           \
    else if (_dx > _hb) _dx1 = _dx - _fb;       \
    else _dx1 = _dx;                            \
    _dist += SQ(_dx1 / _div);                   \
}

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

typedef struct rect_s
{
    int x0, x1, y0, y1;
} rect_s;
typedef struct rect_s *rect;

typedef struct ntuple_list_s
{
    int iter;
    size_t size;
    size_t max_size;
    size_t dim;
    unsigned int *values;
} * ntuple_list;

static void free_ntuple_list(ntuple_list n_tuple)
{
    if (n_tuple == NULL || n_tuple->values == NULL)
        ERROR("free_ntuple_list: invalid n-tuple input.");
    DEALLOC(n_tuple->values);
    DEALLOC(n_tuple);
}

static ntuple_list new_ntuple_list(size_t dim, size_t max_size)
{
    ntuple_list n_tuple;

    /* check parameters */
    if (dim == 0) ERROR("new_ntuple_list: 'dim' must be positive.");

    /* get memory for list structure */
    n_tuple = (ntuple_list)malloc(sizeof(struct ntuple_list_s));
    if (n_tuple == NULL) ERROR("not enough memory.");

    /* initialize list */
    n_tuple->size = 0;
    n_tuple->max_size = max_size;
    n_tuple->dim = dim;

    /* get memory for tuples */
    n_tuple->values = (unsigned int *)malloc(dim * max_size * sizeof(unsigned int));
    if (n_tuple->values == NULL) ERROR("not enough memory.");

    return n_tuple;
}

static void realloc_ntuple_list(ntuple_list n_tuple, size_t max_size)
{
    /* check parameters */
    if (n_tuple == NULL || n_tuple->values == NULL || n_tuple->max_size == 0)
        ERROR("realloc_ntuple_list: invalid n-tuple.");

    /* duplicate number of tuples */
    n_tuple->max_size = max_size;

    /* realloc memory */
    n_tuple->values = REALLOC(n_tuple->values, unsigned int, n_tuple->dim * n_tuple->max_size);
    if (n_tuple->values == NULL) ERROR("not enough memory.");
}

static void add_4tuple(ntuple_list list, float v1, float v2, float v3, float v4)
{
    /* check parameters */
    if (list == NULL) ERROR("add_4tuple: invalid n-tuple input.");
    if (list->dim != 4) ERROR("add_4tuple: the n-tuple must be a 4-tuple.");

    /* if needed, alloc more tuples to 'list' */
    if (list->size == list->max_size) realloc_ntuple_list(list, list->max_size + 1);
    if (list->values == NULL) ERROR("add_4tuple: invalid n-tuple input.");

    /* add new 4-tuple */
    list->values[list->size * list->dim + 0] = v1;
    list->values[list->size * list->dim + 1] = v2;
    list->values[list->size * list->dim + 2] = v3;
    list->values[list->size * list->dim + 3] = v4;

    /* update number of tuples counter */
    list->size++;
}

typedef void (*set_pixel)(void *out, int x, int y, unsigned int val, unsigned int max_val);

static void set_pixel_color(void *out, int x, int y, unsigned int val, unsigned int max_val)
{
    array image = (array)out;
    unsigned int *ptr = image->data + (image->strides[1] * x + image->strides[0] * y) * image->item_size;
    unsigned int new = *ptr + val;
    *ptr = new > max_val ? max_val : new;
}

static void set_pixel_index(void *out, int x, int y, unsigned int val, unsigned int max_val)
{
    ntuple_list idxs = (ntuple_list)out;
    add_4tuple(idxs, (unsigned int)idxs->iter, (unsigned int)x, (unsigned int)y, val > max_val ? max_val : val);
}

/*----------------------------------------------------------------------------*/
/*-------------------------- Bresenham's Algorithm ---------------------------*/
/*----------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------
    Function :  kth_smallest()
    In       :  A 2d line defined by two coordinates (x0, y0, x1, y1)
    Out      :  A rasterized image of the line

    Reference:
        Author: Alois Zingl
        Title: A Rasterizing Algorithm for Drawing Curves
        pdf: http://members.chello.at/%7Eeasyfilter/Bresenham.pdf
        url: http://members.chello.at/~easyfilter/bresenham.html
---------------------------------------------------------------------------*/
static void plot_line_width(void *out, size_t *dims, rect rt, float wd, unsigned int max_val, set_pixel setter, line_profile profile)
{
    /* plot an anti-aliased line of width wd */
    int dx = abs(rt->x1 - rt->x0), sx = rt->x0 < rt->x1 ? 1 : -1;
    int dy = abs(rt->y1 - rt->y0), sy = rt->y0 < rt->y1 ? 1 : -1;
    int err = dx - dy, derr = 0, dx0 = 0, e2, x2, y2, val;    /* error value e_xy */
    float ed = dx + dy == 0 ? 1.0f : sqrtf((float)dx * dx + (float)dy * dy);
    wd = 0.5f * (wd + 1.0f);

    /* define line bounds */
    int wi = roundf(0.5 * wd);
    rect bnd = (rect)malloc(sizeof(struct rect_s));

    if (rt->x0 < rt->x1)
    {
        bnd->x0 = rt->x0 - wi; CLIP(bnd->x0, 0, (int)dims[1] - 1);
        bnd->x1 = rt->x1 + wi; CLIP(bnd->x1, 0, (int)dims[1] - 1);
        err += (rt->x0 - bnd->x0) * dy; rt->x0 = bnd->x0; rt->x1 = bnd->x1;
    }
    else
    {
        bnd->x0 = rt->x1 - wi; CLIP(bnd->x0, 0, (int)dims[1] - 1);
        bnd->x1 = rt->x0 + wi; CLIP(bnd->x1, 0, (int)dims[1] - 1);
        err += (bnd->x1 - rt->x0) * dy; rt->x0 = bnd->x1; rt->x1 = bnd->x0;
    }
    if (rt->y0 < rt->y1)
    {
        bnd->y0 = rt->y0 - wi; CLIP(bnd->y0, 0, (int)dims[0] - 1);
        bnd->y1 = rt->y1 + wi; CLIP(bnd->y1, 0, (int)dims[0] - 1);
        err -= (rt->y0 - bnd->y0) * dx; rt->y0 = bnd->y0; rt->y1 = bnd->y1;
    }
    else
    {
        bnd->y0 = rt->y1 - wi; CLIP(bnd->y0, 0, (int)dims[0] - 1);
        bnd->y1 = rt->y0 + wi; CLIP(bnd->y1, 0, (int)dims[0] - 1);
        err -= (bnd->y1 - rt->y0) * dx; rt->y0 = bnd->y1; rt->y1 = bnd->y0;
    }

    int cnt_max = dx + dy + 4 * wi;
    for (int cnt = 0; cnt < cnt_max; cnt++)
    {
        /* pixel loop */
        err += derr; derr = 0;
        rt->x0 += dx0; dx0 = 0;
        val = profile(max_val, (err - dx + dy) / ed, wd);
        setter(out, rt->x0, rt->y0, val, max_val);

        if (2 * err >= -dx)
        {
            /* x step */
            for (e2 = err + dy, y2 = rt->y0 + sy;
                 abs(e2) < ed * wd && y2 >= bnd->y0 && y2 <= bnd->y1;
                 e2 += dx, y2 += sy)
            {
                val = profile(max_val, e2 / ed, wd);
                setter(out, rt->x0, y2, val, max_val);
            }
            if (rt->x0 == rt->x1) break;
            derr -= dy; dx0 += sx;
        }
        if (2 * err <= dy)
        {
            /* y step */
            for (e2 = err - dx, x2 = rt->x0 + sx;
                 abs(e2) < ed * wd && x2 >= bnd->x0 && x2 <= bnd->x1;
                 e2 -= dy, x2 += sx)
            {
                val = profile(max_val, e2 / ed, wd);
                setter(out, x2, rt->y0, val, max_val);
            }
            if (rt->y0 == rt->y1) break;
            derr += dx; rt->y0 += sy;
        }
    }

    DEALLOC(bnd);
}

int draw_lines(unsigned int *out, size_t Y, size_t X, unsigned int max_val, float *lines, size_t *ldims,
               float dilation, line_profile profile)
{
    /* check parameters */
    if (!out || !lines || !profile) {ERROR("draw_lines: one of the arguments is NULL."); return -1;}
    if (!X && !Y) {ERROR("draw_lines: image size must be positive."); return -1;}

    if (ldims[0] == 0) return 0;

    size_t odims[2] = {Y, X};
    array larr = new_array(2, ldims, sizeof(float), lines);
    array oarr = new_array(2, odims, sizeof(unsigned int), out);

    line ln = init_line(larr, 1);
    rect rt = (rect)malloc(sizeof(struct rect_s));
    float *ln_ptr;

    for (int i = 0; i < (int)ldims[0]; i++)
    {
        UPDATE_LINE(ln, i);
        ln_ptr = ln->data;

        rt->x0 = roundf(ln_ptr[0]); rt->y0 = roundf(ln_ptr[1]);
        rt->x1 = roundf(ln_ptr[2]); rt->y1 = roundf(ln_ptr[3]);

        plot_line_width((void *)oarr, oarr->dims, rt, ln_ptr[4] + dilation, max_val, set_pixel_color, profile);
    }

    DEALLOC(ln); DEALLOC(rt);
    free_array(larr); free_array(oarr);

    return 0;
}

int draw_line_indices(unsigned int **out, size_t *n_idxs, size_t Y, size_t X, unsigned int max_val,
                      float *lines, size_t *ldims, float dilation, line_profile profile)
{
    /* check parameters */
    if (!lines || !profile) {ERROR("draw_line_indices: lines is NULL."); return -1;}
    if (!X && !Y) {ERROR("draw_line_indices: image size must be positive."); return -1;}

    if (ldims[0] == 0) return 0;

    size_t odims[2] = {Y, X};
    array larr = new_array(2, ldims, sizeof(float), lines);
    ntuple_list idxs = new_ntuple_list(4, 1);

    line ln = init_line(larr, 1);
    rect rt = (rect)malloc(sizeof(struct rect_s));
    float *ln_ptr;
    float wd;
    int ln_area;

    for (idxs->iter = 0; idxs->iter < (int)ldims[0]; idxs->iter++)
    {
        UPDATE_LINE(ln, idxs->iter);
        ln_ptr = ln->data;

        rt->x0 = roundf(ln_ptr[0]); rt->y0 = roundf(ln_ptr[1]);
        rt->x1 = roundf(ln_ptr[2]); rt->y1 = roundf(ln_ptr[3]);

        wd = ln_ptr[4] + dilation;
        ln_area = (2 * (int)wd + abs(rt->x1 - rt->x0)) * (2 * (int)wd + abs(rt->y1 - rt->y0));

        // Expand the list if needed
        if (idxs->max_size < idxs->size + (size_t)ln_area)
            realloc_ntuple_list(idxs, idxs->size + (size_t)ln_area);

        // Write down the indices
        plot_line_width((void *)idxs, odims, rt, wd, max_val, set_pixel_index, profile);
    }

    realloc_ntuple_list(idxs, idxs->size);
    *out = idxs->values;
    *n_idxs = idxs->size;

    DEALLOC(ln); DEALLOC(rt);
    free_array(larr); DEALLOC(idxs);

    return 0;
}

static void create_line_image_pair(unsigned int **out, rect orect, size_t Y, size_t X,
    unsigned int max_val, float *ln0, float *ln1, float dilation)
{
    /* Define the bounds 'rt0' of the former line 'ln0' */
    rect rt0 = (rect)malloc(sizeof(struct rect_s));
    rt0->x0 = roundf(ln0[0]); rt0->y0 = roundf(ln0[1]);
    rt0->x1 = roundf(ln0[2]); rt0->y1 = roundf(ln0[3]);

    /* Define the bounds 'rt1' of the latter line 'ln1' */
    rect rt1 = (rect)malloc(sizeof(struct rect_s));
    rt1->x0 = roundf(ln1[0]); rt1->y0 = roundf(ln1[1]);
    rt1->x1 = roundf(ln1[2]); rt1->y1 = roundf(ln1[3]);

    int wi = roundf(MAX(ln0[4], ln1[4]));

    /* Find the outer bounds 'orect' */
    if (rt0->x0 < rt0->x1) {orect->x0 = rt0->x0; orect->x1 = rt0->x1; }
    else {orect->x0 = rt0->x1; orect->x1 = rt0->x0; }
    orect->x0 = MIN(orect->x0, rt1->x0);
    orect->x0 = MIN(orect->x0, rt1->x1);
    orect->x1 = MAX(orect->x1, rt1->x0);
    orect->x1 = MAX(orect->x1, rt1->x1);

    if (rt0->y0 < rt0->y1) {orect->y0 = rt0->y0; orect->y1 = rt0->y1; }
    else {orect->y0 = rt0->y1; orect->y1 = rt0->y0; }
    orect->y0 = MIN(orect->y0, rt1->y0);
    orect->y0 = MIN(orect->y0, rt1->y1);
    orect->y1 = MAX(orect->y1, rt1->y0);
    orect->y1 = MAX(orect->y1, rt1->y1);

    /* Expand the bounds 'orect' by the line's width */
    orect->x0 = MAX(orect->x0 - wi, 0);
    orect->y0 = MAX(orect->y0 - wi, 0);
    orect->x1 = MIN(orect->x1 + wi, (int)X);
    orect->y1 = MIN(orect->y1 + wi, (int)Y);

    /* Create an image 'oarr' */
    size_t odims[2] = {orect->y1 - orect->y0, orect->x1 - orect->x0};
    (*out) = (unsigned int *)calloc(odims[0] * odims[1], sizeof(unsigned int));
    array oarr = new_array(2, odims, sizeof(unsigned int), *out);

    rt0->x0 -= orect->x0; rt0->y0 -= orect->y0; rt0->x1 -= orect->x0; rt0->y1 -= orect->y0;
    rt1->x0 -= orect->x0; rt1->y0 -= orect->y0; rt1->x1 -= orect->x0; rt1->y1 -= orect->y0;

    /* Plot the lines */
    plot_line_width((void *)oarr, oarr->dims, rt0, ln0[4] + dilation, max_val,
                    set_pixel_color, tophat_profile);
    plot_line_width((void *)oarr, oarr->dims, rt1, ln1[4] + dilation, max_val,
                    set_pixel_color, tophat_profile);

    DEALLOC(rt0); DEALLOC(rt1); free_array(oarr);
}

static void create_line_image(unsigned int **out, rect orect, size_t Y, size_t X,
    unsigned int max_val, float *ln, float dilation)
{
    /* Define the bounds 'rt0' of the former line 'ln0' */
    rect rt = (rect)malloc(sizeof(struct rect_s));
    rt->x0 = roundf(ln[0]); rt->y0 = roundf(ln[1]);
    rt->x1 = roundf(ln[2]); rt->y1 = roundf(ln[3]);

    int wi = roundf(ln[4]);

    if (rt->x0 < rt->x1) {orect->x0 = rt->x0; orect->x1 = rt->x1; }
    else {orect->x0 = rt->x1; orect->x1 = rt->x0; }
    if (rt->y0 < rt->y1) {orect->y0 = rt->y0; orect->y1 = rt->y1; }
    else {orect->y0 = rt->y1; orect->y1 = rt->y0; }

    /* Expand the bounds 'orect' by the line's width */
    orect->x0 = MAX(orect->x0 - wi, 0);
    orect->y0 = MAX(orect->y0 - wi, 0);
    orect->x1 = MIN(orect->x1 + wi, (int)X);
    orect->y1 = MIN(orect->y1 + wi, (int)Y);

    /* Create an image 'oarr' */
    size_t odims[2] = {orect->y1 - orect->y0, orect->x1 - orect->x0};
    (*out) = (unsigned int *)calloc(odims[0] * odims[1], sizeof(unsigned int));
    array oarr = new_array(2, odims, sizeof(unsigned int), *out);

    rt->x0 -= orect->x0; rt->y0 -= orect->y0; rt->x1 -= orect->x0; rt->y1 -= orect->y0;

    /* Plot the lines */
    plot_line_width((void *)oarr, oarr->dims, rt, ln[4] + dilation, max_val,
                    set_pixel_color, tophat_profile);

    DEALLOC(rt); free_array(oarr);
}

static float collapse_pair(float *oln, unsigned int *img, rect img_rt, float *data, size_t Y, size_t X,
    float *ln0, float *ln1, size_t lsize)
{
    int j, k;
    unsigned int *img_j, *img_jk;
    float MX, MY, MXY, MXX, MYY, M0;
    float mu_x, mu_y, mu_xy, mu_xx, mu_yy, th, val;
    float *data_j, *data_jk;

    /* Image moments */
    M0 = MX = MY = MXY = MXX = MYY = 0.0f;
    for (j = img_rt->y0, img_j = img, data_j = data + X * img_rt->y0 + img_rt->x0;
         j < img_rt->y1; j++, img_j += img_rt->x1 - img_rt->x0, data_j += X)
        for (k = img_rt->x0, img_jk = img_j, data_jk = data_j;
             k < img_rt->x1; k++, img_jk++, data_jk++)
        {
            val = (*data_jk) * (float)(*img_jk);
            val = (val < 0.0f) ? 0.0f : val;
            M0 += val; MX += k * val; MY += j * val;
            MXY += k * j * val; MYY += j * j * val; MXX += k * k * val;
        }

    if (M0 > 0.0f)
    {
        /* Central moments */
        mu_x = MX / M0; mu_y = MY / M0;
        mu_xy = 2.0f * (MXY / M0 - mu_x * mu_y);
        mu_xx = MXX / M0 - mu_x * mu_x; mu_yy = MYY / M0 - mu_y * mu_y;

        /* Orientation and major axis length */ 
        th = 0.5f * atanf(mu_xy / (mu_xx - mu_yy));
        if (mu_xx < mu_yy) th += M_PI_2;

        /* Collapse the lines */
        float ts[4];
        ts[0] = (ln0[0] - mu_x) * cosf(th) + (ln0[1] - mu_y) * sinf(th);
        ts[1] = (ln1[0] - mu_x) * cosf(th) + (ln1[1] - mu_y) * sinf(th);
        ts[2] = (ln0[2] - mu_x) * cosf(th) + (ln0[3] - mu_y) * sinf(th);
        ts[3] = (ln1[2] - mu_x) * cosf(th) + (ln1[3] - mu_y) * sinf(th);

        float t_min = FLT_MAX, t_max = FLT_MIN;
        for (j = 0; j < 4; j++) if (ts[j] < t_min) t_min = ts[j];
        for (j = 0; j < 4; j++) if (ts[j] > t_max) t_max = ts[j];

        oln[0] = mu_x + t_min * cosf(th);       // x0
        oln[1] = mu_y + t_min * sinf(th);       // y0
        oln[2] = mu_x + t_max * cosf(th);       // x1
        oln[3] = mu_y + t_max * sinf(th);       // y1

        CLIP(oln[0], 0.0f, (float)X); CLIP(oln[2], 0.0f, (float)X);
        CLIP(oln[1], 0.0f, (float)Y); CLIP(oln[3], 0.0f, (float)Y);

        for (j = 4; j < (int)lsize; j++) oln[j] = 0.5f * (ln0[j] + ln1[j]);
    }

    return M0;
}

static float find_overlap(unsigned int *img0, rect rt0, unsigned int *img1, rect rt1, float *data, size_t Y, size_t X)
{
    /* Define overlap rectangle */
    rect rt = (rect)malloc(sizeof(struct rect_s));
    rt->x0 = MIN(rt0->x0, rt1->x0);
    rt->x1 = MAX(rt0->x1, rt1->x1);
    rt->y0 = MIN(rt0->y0, rt1->y0);
    rt->y1 = MAX(rt0->y1, rt1->y1);

    /* image shapes */
    size_t s0[2] = {rt0->y1 - rt0->y0, rt0->x1 - rt0->x0};
    size_t s1[2] = {rt1->y1 - rt1->y0, rt1->x1 - rt1->x0};

    /* calculate overlap and union */
    float ovl = 0.0f, unn = 0.0f;
    int i0, j0, i1, j1;
    unsigned int val0, val1;
    for (int i = rt->y0; i < rt->y1; i++)
    {
        for (int j = rt->x0; j < rt->x1; j++)
        {
            i0 = i - rt0->y0; j0 = j - rt0->x0;
            if ((i0 >= 0) && (i0 < (int)s0[0]) && (j0 >= 0) && (j0 < (int)s0[1])) val0 = img0[i0 * s0[1] + j0];
            else val0 = 0;

            i1 = i - rt1->y0; j1 = j - rt1->x0;
            if ((i1 >= 0) && (i1 < (int)s1[0]) && (j1 >= 0) && (j1 < (int)s1[1])) val1 = img1[i1 * s1[1] + j1];
            else val1 = 0;

            if (val0 && val1) ovl += data[i * X + j];
            if (val0 || val1) unn += data[i * X + j];
        }
    }

    DEALLOC(rt);

    if (unn) return ovl / unn;
    return 0.0f;
}

int filter_lines(float *olines, unsigned char *proc, float *data, size_t Y, size_t X, float *ilines,
    size_t *ldims, float threshold, float dilation)
{
    /* Check parameters */
    if (!olines|| !proc || !ilines) {ERROR("filter_lines: one of the arguments is NULL."); return -1;}
    if (!X && !Y) {ERROR("filter_lines: data array must have a positive size."); return -1;}

    if (ldims[0] == 0) return 0;
    if (*olines != *ilines) memcpy(olines, ilines, ldims[0] * ldims[1] * sizeof(float));

    int i, j, k;
    float M0, val;
    float *ln, *data_j, *data_jk;
    unsigned int *img, *img_j, *img_jk;
    rect rt = (rect)malloc(sizeof(struct rect_s));

    /* Filter faint lines */
    for (i = 0, ln = olines; i < (int)ldims[0]; i++, ln += ldims[1])
    {
        /* check if processed already */
        if (proc[i] == 0) continue;

        /* Draw a line */
        create_line_image(&img, rt, Y, X, 1, ln, dilation);

        /* Calculate zero-th image moment */
        M0 = 0.0f;
        for (j = rt->y0, img_j = img, data_j = data + X * rt->y0 + rt->x0;
             j < rt->y1; j++, img_j += rt->x1 - rt->x0, data_j += X)
            for (k = rt->x0, img_jk = img_j, data_jk = data_j;
                k < rt->x1; k++, img_jk++, data_jk++)
            {
                val = (*data_jk) * (float)(*img_jk);
                M0 += (val < 0.0f) ? 0.0f : val;
            }

        /* Delete the line if M0 is too small */
        if (M0 < threshold)
        {
            memset(ln, 0, ldims[1] * sizeof(float));
            proc[i] = 0;
        }

        DEALLOC(img);
    }

    DEALLOC(rt);

    return 0;
}

int group_lines(float *olines, unsigned char *proc, float *data, size_t Y, size_t X, float *ilines, size_t *ldims,
    float cutoff, float threshold, float dilation)
{
    /* Check parameters */
    if (!olines|| !data || !ilines) {ERROR("group_lines: one of the arguments is NULL."); return -1;}
    if (!X && !Y) {ERROR("group_lines: data array must have a positive size."); return -1;}
    
    if (ldims[0] == 0) return 0;
    if (*olines != *ilines) memcpy(olines, ilines, ldims[0] * ldims[1] * sizeof(float));

    int i, j, n_pairs = 0;
    float dist, corr, M0;
    float *ln0, *ln1;

    /* Find the closest line */
    size_t max_pairs = ldims[0];
    unsigned int *pairs = MALLOC(unsigned int, 2 * max_pairs);
    float *ds = MALLOC(float, max_pairs);
    for (i = 0, ln0 = olines; i < (int)ldims[0]; i++, ln0 += ldims[1])
    {
        /* check if processed already */
        if (proc[i] == 0) continue;

        for (j = 0, ln1 = olines; j < (int)ldims[0]; j++, ln1 += ldims[1])
        {
            /* check if processed already */
            if (i == j || proc[j] == 0) continue;

            /* calculate the distance between two lines */
            dist = 0.5f * sqrtf(SQ(ln0[0] + ln0[2] - ln1[0] - ln1[2]) +
                                SQ(ln0[1] + ln0[3] - ln1[1] - ln1[3]));

            /* if the pair of line is close enough, add to the list of pairs */
            if (dist < cutoff)
            {
                pairs[2 * n_pairs] = i;
                pairs[2 * n_pairs + 1] = j;
                ds[n_pairs] = dist; n_pairs++;
            }

            /* expand the list if necessary */
            if ((int)max_pairs == n_pairs)
            {
                max_pairs += ldims[0];
                pairs = REALLOC(pairs, unsigned int, 2 * max_pairs);
                ds = REALLOC(ds, float, max_pairs);
            }
        }
    }

    /* Exit if no pairs have been found */
    if (!n_pairs) {DEALLOC(pairs); DEALLOC(ds); return 0;}

    size_t *inds = MALLOC(size_t, n_pairs);
    for (i = 0; i < n_pairs; i++) inds[i] = i;

    /* Sort the pairs based on the distance */
    qsort_r(inds, n_pairs, sizeof(size_t), indirect_compare_float, ds);
    DEALLOC(ds);

    float *oln = MALLOC(float, ldims[1]);
    unsigned int *img_pair, *img_oln;
    rect rt_pair = (rect)malloc(sizeof(struct rect_s));
    rect ort = (rect)malloc(sizeof(struct rect_s));

    /* Collapse the pairs */
    for (i = 0; i < n_pairs; i++)
    {

        ln0 = olines + ldims[1] * pairs[2 * inds[i]];
        ln1 = olines + ldims[1] * pairs[2 * inds[i] + 1];

        if (proc[pairs[2 * inds[i]]] && proc[pairs[2 * inds[i] + 1]])
        {
            /* Create an image of a pair */
            create_line_image_pair(&img_pair, rt_pair, Y, X, 1, ln0, ln1, dilation);
            
            /* Collapse a pair of lines */
            M0 = collapse_pair(oln, img_pair, rt_pair, data, Y, X, ln0, ln1, ldims[1]);

            if (M0 > 0.0f)
            {
                /* Create an image of oln */
                create_line_image(&img_oln, ort, Y, X, 1, oln, dilation);

                /* Find overlap between imp_pair and imp_oln */
                corr = find_overlap(img_pair, rt_pair, img_oln, ort, data, Y, X);

                if (corr > threshold)
                {
                    memcpy(ln0, oln, ldims[1] * sizeof(float));
                    memset(ln1, 0, ldims[1] * sizeof(float));
                    proc[pairs[2 * inds[i] + 1]] = 0;
                }

                DEALLOC(img_oln);
            }

            DEALLOC(img_pair);
        }
    }

    DEALLOC(pairs);
    DEALLOC(rt_pair); DEALLOC(ort); DEALLOC(oln);

    return 0;
}

/*----------------------------------------------------------------------------*/
/*------------------------------- Euler angles -------------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/*  Euler angles with Bunge convention

        ang = [phi1, Phi, phi2]
        phi1 \el [0, 2 * M_PI)
        Phi  \el [0, M_PI)
        phi2 \el [0, 2 * M_PI)

    See the following article for more info:
    http://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf
 */

static void rotmat_to_euler(double *ang, double *rm)
{
    ang[1] = acos(rm[8]);
    if (ang[1] < 1e-8) {ang[0] = atan2(-rm[3], rm[0]); ang[2] = 0.0; }
    else if (M_PI - ang[1] < TOL)
    {ang[0] = atan2(rm[3], rm[0]); ang[2] = 0.0; }
    else {ang[0] = atan2(rm[6], -rm[7]); ang[2] = atan2(rm[2], rm[5]); }
    if (ang[0] < 0.0) ang[0] += M_2__PI;
    if (ang[2] < 0.0) ang[2] += M_2__PI;
}

int compute_euler_angles(double *angles, double *rot_mats, size_t n_mats)
{
    /* check parameters */
    if (!angles || !rot_mats) {ERROR("compute_euler_angles: one of the arguments is NULL."); return -1;}
    if (!n_mats) {ERROR("compute_euler_angles: number of matrices must be positive."); return -1;}   

    size_t rm_dims[2] = {n_mats, 9};
    array rm_arr = new_array(2, rm_dims, sizeof(double), rot_mats);
    line rm_ln = init_line(rm_arr, 1);

    size_t a_dims[2] = {n_mats, 3};
    array ang_arr = new_array(2, a_dims, sizeof(double), angles);
    line ang_ln = init_line(ang_arr, 1);

    for (int i = 0; i < (int)n_mats; i++)
    {
        UPDATE_LINE(rm_ln, i);
        UPDATE_LINE(ang_ln, i);

        rotmat_to_euler(ang_ln->data, rm_ln->data);
    }

    DEALLOC(ang_ln); DEALLOC(rm_ln);
    free_array(ang_arr); free_array(rm_arr);

    return 0;
}

static void euler_to_rotmat(double *rm, double *ang)
{
    double c0 = cos(ang[0]), c1 = cos(ang[1]), c2 = cos(ang[2]);
    double s0 = sin(ang[0]), s1 = sin(ang[1]), s2 = sin(ang[2]);

    rm[0] = c0 * c2 - s0 * s2 * c1;
    rm[1] = s0 * c2 + c0 * s2 * c1;
    rm[2] = s2 * s1;
    rm[3] = -c0 * s2 - s0 * c2 * c1;
    rm[4] = -s0 * s2 + c0 * c2 * c1;
    rm[5] = c2 * s1;
    rm[6] = s0 * s1;
    rm[7] = -c0 * s1;
    rm[8] = c1;
}

int compute_euler_matrix(double *rot_mats, double *angles, size_t n_mats)
{
    /* check parameters */
    if (!angles || !rot_mats) {ERROR("compute_euler_matrix: one of the arguments is NULL."); return -1;}
    if (!n_mats) {ERROR("compute_euler_matrix: number of matrices must be positive."); return -1;}   

    size_t rm_dims[2] = {n_mats, 9};
    array rm_arr = new_array(2, rm_dims, sizeof(double), rot_mats);
    line rm_ln = init_line(rm_arr, 1);

    size_t a_dims[2] = {n_mats, 3};
    array ang_arr = new_array(2, a_dims, sizeof(double), angles);
    line ang_ln = init_line(ang_arr, 1);

    for (int i = 0; i < (int)n_mats; i++)
    {
        UPDATE_LINE(rm_ln, i);
        UPDATE_LINE(ang_ln, i);

        euler_to_rotmat(rm_ln->data, ang_ln->data);
    }

    DEALLOC(ang_ln); DEALLOC(rm_ln);
    free_array(ang_arr); free_array(rm_arr);

    return 0;
}

/*----------------------------------------------------------------------------*/
/*  Tilt around an axis

        ang = [theta, alpha, beta]
        theta \el [0, 2 * M_PI)         Angle of rotation
        alpha  \el [0, M_PI)            Angle between the axis of rotation and 0Z
        phi \el [0, 2 * M_PI)           Polar angle of the axis of rotation
 */

static void rotmat_to_tilt(double *ang, double *rm)
{
    double a0 = rm[7] - rm[5];
    double a1 = rm[2] - rm[6];
    double a2 = rm[3] - rm[1];
    double l = sqrt(SQ(a0) + SQ(a1) + SQ(a2));
    ang[0] = acos(0.5 * (rm[0] + rm[4] + rm[8] - 1.0));
    ang[1] = acos(a2 / l);
    ang[2] = atan2(a1, a0);
}

int compute_tilt_angles(double *angles, double *rot_mats, size_t n_mats)
{
    /* check parameters */
    if (!angles || !rot_mats) {ERROR("compute_tilt_angles: one of the arguments is NULL."); return -1;}
    if (!n_mats) {ERROR("compute_tilt_angles: number of matrices must be positive."); return -1;}   

    size_t rm_dims[2] = {n_mats, 9};
    array rm_arr = new_array(2, rm_dims, sizeof(double), rot_mats);
    line rm_ln = init_line(rm_arr, 1);

    size_t a_dims[2] = {n_mats, 3};
    array ang_arr = new_array(2, a_dims, sizeof(double), angles);
    line ang_ln = init_line(ang_arr, 1);

    for (int i = 0; i < (int)n_mats; i++)
    {
        UPDATE_LINE(rm_ln, i);
        UPDATE_LINE(ang_ln, i);

        rotmat_to_tilt(ang_ln->data, rm_ln->data);
    }

    DEALLOC(ang_ln); DEALLOC(rm_ln);
    free_array(ang_arr); free_array(rm_arr);

    return 0;
}


static void tilt_to_rotmat(double *rm, double *ang)
{
    float a = cos(0.5 * ang[0]), b = -sin(ang[1]) * cos(ang[2]) * sin(0.5 * ang[0]);
    float c = -sin(ang[1]) * sin(ang[2]) * sin(0.5 * ang[0]), d = -cos(ang[1]) * sin(0.5 * ang[0]);

    rm[0] = a * a + b * b - c * c - d * d;
    rm[1] = 2.0 * (b * c + a * d);
    rm[2] = 2.0 * (b * d - a * c);
    rm[3] = 2.0 * (b * c - a * d);
    rm[4] = a * a + c * c - b * b - d * d;
    rm[5] = 2.0 * (c * d + a * b);
    rm[6] = 2.0 * (b * d + a * c);
    rm[7] = 2.0 * (c * d - a * b);
    rm[8] = a * a + d * d - b * b - c * c;
}

int compute_tilt_matrix(double *rot_mats, double *angles, size_t n_mats)
{
    /* check parameters */
    if (!angles || !rot_mats) {ERROR("compute_tilt_matrix: one of the arguments is NULL."); return -1;}
    if (!n_mats) {ERROR("compute_tilt_matrix: number of matrices must be positive."); return -1;}   

    size_t rm_dims[2] = {n_mats, 9};
    array rm_arr = new_array(2, rm_dims, sizeof(double), rot_mats);
    line rm_ln = init_line(rm_arr, 1);

    size_t a_dims[2] = {n_mats, 3};
    array ang_arr = new_array(2, a_dims, sizeof(double), angles);
    line ang_ln = init_line(ang_arr, 1);

    for (int i = 0; i < (int)n_mats; i++)
    {
        UPDATE_LINE(rm_ln, i);
        UPDATE_LINE(ang_ln, i);

        tilt_to_rotmat(rm_ln->data, ang_ln->data);
    }

    DEALLOC(ang_ln); DEALLOC(rm_ln);
    free_array(ang_arr); free_array(rm_arr);

    return 0;
}

/*-------------------------------------------------------------------------------*/
/** Calculate the rotation matrix rm, that rotates unit vector a to unit vector b.

    @param rm       Output rotation matrix.
    @param a        Unit vector a.
    @param b        Unit vector b.

    @note           Yields nan, if cos(a, b) = -1.0.
 */
static void rotation_of_a_to_b(double *rm, double *a, double *b)
{
    double vx = a[1] * b[2] - a[2] * b[1];
    double vy = a[2] * b[0] - a[0] * b[2];
    double vz = a[0] * b[1] - a[1] * b[0];
    double rat = 1.0 / (1.0 + a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);
    rm[0] = 1.0 - rat * (SQ(vy) + SQ(vz));
    rm[1] = -vz + rat * vx * vy;
    rm[2] =  vy + rat * vx * vz;
    rm[3] =  vz + rat * vx * vy;
    rm[4] = 1.0 - rat * (SQ(vx) + SQ(vz));
    rm[5] = -vx + rat * vy * vz;
    rm[6] = -vy + rat * vx * vz;
    rm[7] =  vx + rat * vy * vz;
    rm[8] = 1.0 - rat * (SQ(vx) + SQ(vy));
}

int compute_rotations(double *rot_mats, double *as, double *bs, size_t n_mats)
{
    /* check parameters */
    if (!as || !bs || !rot_mats) {ERROR("compute_rotations: one of the arguments is NULL."); return -1;}
    if (!n_mats) {ERROR("compute_rotations: number of matrices must be positive."); return -1;}   

    size_t rm_dims[2] = {n_mats, 9};
    array rm_arr = new_array(2, rm_dims, sizeof(double), rot_mats);
    line rm_ln = init_line(rm_arr, 1);
    double a[3], b[3];
    double a_abs, b_abs;

    for (int i = 0; i < (int)n_mats; i++)
    {
        UPDATE_LINE(rm_ln, i);

        a_abs = sqrt(SQ(as[3 * i]) + SQ(as[3 * i + 1]) + SQ(as[3 * i + 2]));
        a[0] = as[3 * i] / a_abs; a[1] = as[3 * i + 1] / a_abs; a[2] = as[3 * i + 2] / a_abs;
        b_abs = sqrt(SQ(bs[3 * i]) + SQ(bs[3 * i + 1]) + SQ(bs[3 * i + 2]));
        b[0] = bs[3 * i] / b_abs; b[1] = bs[3 * i + 1] / b_abs; b[2] = bs[3 * i + 2] / b_abs;
        rotation_of_a_to_b(rm_ln->data, a, b);
    }

    DEALLOC(rm_ln); free_array(rm_arr);

    return 0;
}