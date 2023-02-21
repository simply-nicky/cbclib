#include "img_proc.h"
#include "array.h"
#include "kd_tree.h"

#define TOL 3.1425926535897937e-05

#define WRAP_DIST(_dist, _dx, _hb, _fb, _div)                   \
    do {float _dx1; if ((_dx) < -(_hb)) _dx1 = (_dx) + (_fb);   \
        else if ((_dx) > (_hb)) _dx1 = (_dx) - (_fb);           \
        else _dx1 = (_dx); (_dist) += SQ(_dx1 / _div); } while (0)

typedef struct index_tuple_s
{
    int iter;
    size_t size;
    size_t max_size;
    unsigned *idx, *x, *y;
    float *val;
} index_tuple_s;
typedef struct index_tuple_s *index_tuple;

static void free_index_tuple(index_tuple itup)
{
    if (itup == NULL || itup->idx == NULL || itup->x == NULL || itup->y == NULL ||
        itup->val == NULL) ERROR("free_index_tuple: invalid index-tuple input.");
    DEALLOC(itup->idx); DEALLOC(itup->x); DEALLOC(itup->y); DEALLOC(itup->val);
    DEALLOC(itup);
}

static index_tuple new_index_tuple(size_t max_size)
{
    index_tuple itup;

    /* get memory for list structure */
    itup = (index_tuple)malloc(sizeof(struct index_tuple_s));

    /* initialize list */
    itup->size = 0;
    itup->max_size = max_size;

    /* get memory for tuples */
    itup->idx = MALLOC(unsigned, itup->max_size);
    itup->x = MALLOC(unsigned, itup->max_size);
    itup->y = MALLOC(unsigned, itup->max_size);
    itup->val = MALLOC(float, itup->max_size);

    return itup;
}

static void realloc_index_tuple(index_tuple itup, size_t max_size)
{
    /* check parameters */
    if (itup == NULL || itup->idx == NULL || itup->x == NULL || itup->y == NULL ||
        itup->val == NULL || itup->max_size == 0)
        ERROR("realloc_index_tuple: invalid index-tuple.");

    /* duplicate number of tuples */
    itup->max_size = max_size;

    /* realloc memory */
    itup->idx = REALLOC(itup->idx, unsigned, itup->max_size);
    itup->x = REALLOC(itup->x, unsigned, itup->max_size);
    itup->y = REALLOC(itup->y, unsigned, itup->max_size);
    itup->val = REALLOC(itup->val, float, itup->max_size);
}

static void add_index_tuple(index_tuple itup, unsigned idx, unsigned x, unsigned y, float val)
{
    /* check parameters */
    if (itup == NULL || itup->idx == NULL || itup->x == NULL || itup->y == NULL ||
        itup->val == NULL) ERROR("add_index_tuple: invalid index-tuple input.");

    /* if needed, alloc more tuples to 'itup' */
    if (itup->size == itup->max_size) realloc_index_tuple(itup, itup->max_size + 1);

    /* add new 4-tuple */
    itup->idx[itup->size] = idx; itup->x[itup->size] = x;
    itup->y[itup->size] = y; itup->val[itup->size] = val;

    /* update number of tuples counter */
    itup->size++;
}

typedef void (*set_pixel)(void *out, int x, int y, float val);

static void set_pixel_int(void *out, int x, int y, float val)
{
    array image = (array)out;
    unsigned *ptr = image->data + (image->strides[1] * x + image->strides[0] * y) * image->item_size;
    unsigned new = MAX(*ptr, (unsigned)val);
    *ptr = new;
}

static void set_pixel_float(void *out, int x, int y, float val)
{
    array image = (array)out;
    float *ptr = image->data + (image->strides[1] * x + image->strides[0] * y) * image->item_size;
    float new = MAX(*ptr, val);
    *ptr = new;
}

static void set_pixel_index(void *out, int x, int y, float val)
{
    index_tuple idxs = (index_tuple)out;
    add_index_tuple(idxs, idxs->iter, x, y, val);
}

/* Points (integer) follow the convention:      [k, j, i], where {i <-> x, j <-> y, k <-> z}
   Coordinates (float) follow the convention:   [x, y, z]
 */

/*----------------------------------------------------------------------------*/
/*-------------------------- Bresenham's Algorithm ---------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------
    Function :  plot_line_width()
    In       :  A 2d line defined by two (integer) points (j0, i0, j1, i1)
                and a (float) width wd
    Out      :  A rasterized image of the line

    Reference:
        Author: Alois Zingl
        Title: A Rasterizing Algorithm for Drawing Curves
        pdf: http://members.chello.at/%7Eeasyfilter/Bresenham.pdf
        url: http://members.chello.at/~easyfilter/bresenham.html
------------------------------------------------------------------------------*/
static void plot_line_width(void *out, const size_t *dims, int *points, float wd, unsigned max_val,
                            set_pixel setter, line_profile profile)
{
    /* create a volatile copy of the input line */
    /* the points are given by [j0, i0, j1, i1] */
    int pts[4] = {points[0], points[1], points[2], points[3]};

    /* plot an anti-aliased line of width wd */
    int dx = abs(pts[3] - pts[1]), sx = pts[1] < pts[3] ? 1 : -1;
    int dy = abs(pts[2] - pts[0]), sy = pts[0] < pts[2] ? 1 : -1;
    int err = dx - dy, derr = 0, dx0 = 0, e2, x2, y2;    /* error value e_xy */
    float ed = (dx + dy == 0) ? 1.0f : sqrtf((float)dx * dx + (float)dy * dy), val;
    wd = 0.5f * (wd + 1.0f);

    /* define image bounds */
    int wi = roundf(0.5 * wd);
    int bnd[4];

    if (pts[1] < pts[3])
    {
        bnd[1] = pts[1] - wi; CLIP(bnd[1], 0, (int)dims[1] - 1);
        bnd[3] = pts[3] + wi; CLIP(bnd[3], 0, (int)dims[1] - 1);
        err += (pts[1] - bnd[1]) * dy; pts[1] = bnd[1]; pts[3] = bnd[3];
    }
    else
    {
        bnd[1] = pts[3] - wi; CLIP(bnd[1], 0, (int)dims[1] - 1);
        bnd[3] = pts[1] + wi; CLIP(bnd[3], 0, (int)dims[1] - 1);
        err += (bnd[3] - pts[1]) * dy; pts[1] = bnd[3]; pts[3] = bnd[1];
    }
    if (pts[0] < pts[2])
    {
        bnd[0] = pts[0] - wi; CLIP(bnd[0], 0, (int)dims[0] - 1);
        bnd[2] = pts[2] + wi; CLIP(bnd[2], 0, (int)dims[0] - 1);
        err -= (pts[0] - bnd[0]) * dx; pts[0] = bnd[0]; pts[2] = bnd[2];
    }
    else
    {
        bnd[0] = pts[2] - wi; CLIP(bnd[0], 0, (int)dims[0] - 1);
        bnd[2] = pts[0] + wi; CLIP(bnd[2], 0, (int)dims[0] - 1);
        err -= (bnd[2] - pts[0]) * dx; pts[0] = bnd[2]; pts[2] = bnd[0];
    }

    /* Main loop */
    int cnt_max = dx + dy + 4 * wi;
    for (int cnt = 0; cnt < cnt_max; cnt++)
    {
        /* pixel loop */
        err += derr; derr = 0;
        pts[1] += dx0; dx0 = 0;
        val = max_val * profile((err - dx + dy) / ed, wd);
        setter(out, pts[1], pts[0], val);

        if (2 * err >= -dx)
        {
            /* x step */
            for (e2 = err + dy, y2 = pts[0] + sy;
                 abs(e2) < ed * wd && y2 >= bnd[0] && y2 <= bnd[2];
                 e2 += dx, y2 += sy)
            {
                val = max_val * profile(e2 / ed, wd);
                setter(out, pts[1], y2, val);
            }
            if (pts[1] == pts[3]) break;
            derr -= dy; dx0 += sx;
        }
        if (2 * err <= dy)
        {
            /* y step */
            for (e2 = err - dx, x2 = pts[1] + sx;
                 abs(e2) < ed * wd && x2 >= bnd[1] && x2 <= bnd[3];
                 e2 -= dy, x2 += sx)
            {
                val = max_val * profile(e2 / ed, wd);
                setter(out, x2, pts[0], val);
            }
            if (pts[0] == pts[2]) break;
            derr += dx; pts[0] += sy;
        }
    }
}

int draw_line_int(unsigned *out, const size_t *dims, unsigned max_val, float *lines, const size_t *ldims,
                 float dilation, line_profile profile)
{
    /* check parameters */
    if (!out || !lines || !profile) {ERROR("draw_line: one of the arguments is NULL."); return -1;}
    if (!dims[0] && !dims[1]) {ERROR("draw_line: image size must be positive."); return -1;}
    if (dilation < 0.0) {ERROR("draw_line: dilation must be a positive number"); return -1;}

    if (ldims[0] == 0) return 0;

    array larr = new_array(2, ldims, sizeof(float), lines);
    array oarr = new_array(2, dims, sizeof(unsigned), out);
    line ln = init_line(larr, 1);

    for (int i = 0; i < (int)ldims[0]; i++)
    {
        UPDATE_LINE(ln, i);

        /* Convert coordinates to points */
        int pts[4] = {roundf(GET(ln, float, 1)), roundf(GET(ln, float, 0)),
                      roundf(GET(ln, float, 3)), roundf(GET(ln, float, 2))};

        plot_line_width(oarr, oarr->dims, pts, GET(ln, float, 4) + dilation, max_val,
                        set_pixel_int, profile);
    }

    DEALLOC(ln); free_array(larr); free_array(oarr);

    return 0;
}

int draw_line_float(float *out, const size_t *dims, float *lines, const size_t *ldims, float dilation,
                    line_profile profile)
{
    /* check parameters */
    if (!out || !lines || !profile) {ERROR("draw_line: one of the arguments is NULL."); return -1;}
    if (!dims[0] && !dims[1]) {ERROR("draw_line: image size must be positive."); return -1;}
    if (dilation < 0.0) {ERROR("draw_line: dilation must be a positive number"); return -1;}

    if (ldims[0] == 0) return 0;

    array larr = new_array(2, ldims, sizeof(float), lines);
    array oarr = new_array(2, dims, sizeof(float), out);
    line ln = init_line(larr, 1);

    for (int i = 0; i < (int)ldims[0]; i++)
    {
        UPDATE_LINE(ln, i);

        /* Convert coordinates to points */
        int pts[4] = {roundf(GET(ln, float, 1)), roundf(GET(ln, float, 0)),
                      roundf(GET(ln, float, 3)), roundf(GET(ln, float, 2))};

        plot_line_width(oarr, oarr->dims, pts, GET(ln, float, 4) + dilation, 1,
                        set_pixel_float, profile);
    }

    DEALLOC(ln); free_array(larr); free_array(oarr);

    return 0;
}

int draw_line_index(unsigned **idx, unsigned **x, unsigned **y, float **val, size_t *n_idxs, const size_t *dims,
                    float *lines, const size_t *ldims, float dilation, line_profile profile)
{
    /* check parameters */
    if (!lines || !profile) {ERROR("draw_line_index: lines is NULL."); return -1;}
    if (!dims[0] && !dims[1]) {ERROR("draw_line_index: image size must be positive."); return -1;}
    if (dilation < 0.0) {ERROR("draw_line_index: dilation must be a positive number"); return -1;}

    if (ldims[0] == 0) return 0;

    array larr = new_array(2, ldims, sizeof(float), lines);
    index_tuple idxs = new_index_tuple(1);

    line ln = init_line(larr, 1);
    float wd;
    int ln_area;

    for (idxs->iter = 0; idxs->iter < (int)ldims[0]; idxs->iter++)
    {
        UPDATE_LINE(ln, idxs->iter);

        /* Convert coordinates to points */
        int pts[4] = {roundf(GET(ln, float, 1)), roundf(GET(ln, float, 0)),
                      roundf(GET(ln, float, 3)), roundf(GET(ln, float, 2))};

        wd = GET(ln, float, 4) + dilation;
        ln_area = (2 * (int)wd + abs(pts[2] - pts[0])) * (2 * (int)wd + abs(pts[3] - pts[1]));

        // Expand the list if needed
        if (idxs->max_size < idxs->size + (size_t)ln_area)
            realloc_index_tuple(idxs, idxs->size + (size_t)ln_area);

        // Write down the indices
        plot_line_width(idxs, dims, pts, wd, 1, set_pixel_index, profile);
    }

    realloc_index_tuple(idxs, idxs->size);
    *idx = idxs->idx; *x = idxs->x; *y = idxs->y; *val = idxs->val;
    *n_idxs = idxs->size;

    DEALLOC(ln); free_array(larr); DEALLOC(idxs);

    return 0;
}

static void create_line_image_pair(float **out, int **pts, const size_t *dims, float *ln0, float *ln1,
                                   float dilation, line_profile profile)
{
    /* Convert coordinates 'ln0' to points */
    int pts0[4] = {roundf(ln0[1]), roundf(ln0[0]), roundf(ln0[3]), roundf(ln0[2])};

    /* Convert coordinates 'ln1' to points */
    int pts1[4] = {roundf(ln1[1]), roundf(ln1[0]), roundf(ln1[3]), roundf(ln1[2])};

    int wi = roundf(MAX(ln0[4], ln1[4]));

    /* Find the outer bounds 'orect' */
    if (pts0[0] < pts0[2])
    {
        (*pts)[0] = pts0[0]; (*pts)[2] = pts0[2];
    }
    else
    {
        (*pts)[0] = pts0[2]; (*pts)[2] = pts0[0];
    }
    (*pts)[0] = MIN((*pts)[0], pts1[0]);
    (*pts)[0] = MIN((*pts)[0], pts1[2]);
    (*pts)[2] = MAX((*pts)[2], pts1[0]);
    (*pts)[2] = MAX((*pts)[2], pts1[2]);

    if (pts0[1] < pts0[3])
    {
        (*pts)[1] = pts0[1]; (*pts)[3] = pts0[3];
    }
    else
    {
        (*pts)[1] = pts0[3]; (*pts)[3] = pts0[1];
    }
    (*pts)[1] = MIN((*pts)[1], pts1[1]);
    (*pts)[1] = MIN((*pts)[1], pts1[3]);
    (*pts)[3] = MAX((*pts)[3], pts1[1]);
    (*pts)[3] = MAX((*pts)[3], pts1[3]);

    /* Expand the bounds 'orect' by the line's width */
    (*pts)[0] = MAX((*pts)[0] - wi, 0);
    (*pts)[1] = MAX((*pts)[1] - wi, 0);
    (*pts)[2] = MIN((*pts)[2] + wi, (int)dims[0]);
    (*pts)[3] = MIN((*pts)[3] + wi, (int)dims[1]);

    /* Create an image 'oarr' */
    size_t odims[2] = {(*pts)[2] - (*pts)[0], (*pts)[3] - (*pts)[1]};
    float *img1 = calloc(odims[0] * odims[1], sizeof(float));
    float *img2 = calloc(odims[0] * odims[1], sizeof(float));
    array arr1 = new_array(2, odims, sizeof(float), img1);
    array arr2 = new_array(2, odims, sizeof(float), img2);

    /* Offset points 'pts0', 'pts1' by the origin of 'orect' */
    pts0[0] -= (*pts)[0]; pts0[1] -= (*pts)[1];
    pts0[2] -= (*pts)[0]; pts0[3] -= (*pts)[1];
    pts1[0] -= (*pts)[0]; pts1[1] -= (*pts)[1];
    pts1[2] -= (*pts)[0]; pts1[3] -= (*pts)[1];

    /* Plot the lines */
    plot_line_width(arr1, arr1->dims, pts0, ln0[4] + dilation, 1, set_pixel_float, profile);
    plot_line_width(arr2, arr2->dims, pts1, ln1[4] + dilation, 1, set_pixel_float, profile);

    free_array(arr1); free_array(arr2);

    (*out) = MALLOC(float, odims[0] * odims[1]);
    for (int i = 0; i < (int)(odims[0] * odims[1]); i++) (*out)[i] = MAX(img1[i], img2[i]);

    DEALLOC(img1); DEALLOC(img2);
}

static void create_line_image(float **out, int **opts, const size_t *dims, float *ln, float dilation,
                              line_profile profile)
{
    /* Convert coordinates 'ln' to points */
    int pts[4] = {roundf(ln[1]), roundf(ln[0]), roundf(ln[3]), roundf(ln[2])};

    int wi = roundf(ln[4]);

    /* Define the image bounds 'opts' */
    if (pts[0] < pts[2])
    {
        (*opts)[0] = pts[0]; (*opts)[2] = pts[2];
    }
    else
    {
        (*opts)[0] = pts[2]; (*opts)[2] = pts[0];
    }
    if (pts[1] < pts[3])
    {
        (*opts)[1] = pts[1]; (*opts)[3] = pts[3];
    }
    else
    {
        (*opts)[1] = pts[3]; (*opts)[3] = pts[1];
    }

    /* Expand the bounds 'opts' by the line's width */
    (*opts)[0] = MAX((*opts)[0] - wi, 0);
    (*opts)[1] = MAX((*opts)[1] - wi, 0);
    (*opts)[2] = MIN((*opts)[2] + wi, (int)dims[0]);
    (*opts)[3] = MIN((*opts)[3] + wi, (int)dims[1]);

    /* Create an image 'oarr' */
    size_t odims[2] = {(*opts)[2] - (*opts)[0], (*opts)[3] - (*opts)[1]};
    (*out) = (float *)calloc(odims[0] * odims[1], sizeof(float));
    array oarr = new_array(2, odims, sizeof(float), *out);

    /* Offset points 'pts' by the origin of 'opts' */
    pts[0] -= (*opts)[0]; pts[1] -= (*opts)[1];
    pts[2] -= (*opts)[0]; pts[3] -= (*opts)[1];

    /* Plot the lines */
    plot_line_width(oarr, oarr->dims, pts, ln[4] + dilation, 1, set_pixel_float, profile);

    free_array(oarr);
}

static float collapse_pair(float *oln, float *img, int *pts, float *data, const size_t *dims,
                           float *ln0, float *ln1, size_t lsize)
{
    int i, idx, pt[2];
    float MX, MY, MXY, MXX, MYY, M0;
    float mu_x, mu_y, mu_xy, mu_xx, mu_yy, th, val;
    /* Create data array */
    array darr = new_array(2, dims, sizeof(float), data);
    rect_iter ri;

    /* Calculate image moments */
    M0 = MX = MY = MXY = MXX = MYY = 0.0f;
    for (ri = ri_ini(2, pts, pts + 2); !ri_end(ri); ri_inc(ri))
    {
        /* point is given by [j, i], where {j <-> y, i <-> x} */
        pt[0] = ri->coord[0] + pts[0];
        pt[1] = ri->coord[1] + pts[1];
        RAVEL_INDEX(pt, &idx, darr);

        val = data[idx] * img[ri->index];
        val = (val < 0.0f) ? 0.0f : val;
        M0 += val; MX += pt[1] * val; MY += pt[0] * val;
        MXY += pt[0] * pt[1] * val; MYY += SQ(pt[0]) * val; MXX += SQ(pt[1]) * val;
    }
    ri_del(ri);

    if (M0 > 0.0f)
    {
        /* Central moments */
        mu_x = MX / M0; mu_y = MY / M0;
        mu_xy = 2.0f * (MXY / M0 - mu_x * mu_y);
        mu_xx = MXX / M0 - mu_x * mu_x; mu_yy = MYY / M0 - mu_y * mu_y;

        /* Line inclination */ 
        th = 0.5f * atanf(mu_xy / (mu_xx - mu_yy));
        if (mu_xx < mu_yy) th += M_PI_2;

        /* Find the bounds of 'ln0' and 'ln1' */
        float ts[4];
        ts[0] = (ln0[0] - mu_x) * cosf(th) + (ln0[1] - mu_y) * sinf(th);
        ts[1] = (ln1[0] - mu_x) * cosf(th) + (ln1[1] - mu_y) * sinf(th);
        ts[2] = (ln0[2] - mu_x) * cosf(th) + (ln0[3] - mu_y) * sinf(th);
        ts[3] = (ln1[2] - mu_x) * cosf(th) + (ln1[3] - mu_y) * sinf(th);

        /* Find the extreme line bounds t_min, t_max */
        float t_min = FLT_MAX, t_max = FLT_MIN;
        for (i = 0; i < 4; i++) if (ts[i] < t_min) t_min = ts[i];
        for (i = 0; i < 4; i++) if (ts[i] > t_max) t_max = ts[i];

        oln[0] = mu_x + t_min * cosf(th);       // x0
        oln[1] = mu_y + t_min * sinf(th);       // y0
        oln[2] = mu_x + t_max * cosf(th);       // x1
        oln[3] = mu_y + t_max * sinf(th);       // y1

        CLIP(oln[0], 0.0f, (float)dims[1]); CLIP(oln[2], 0.0f, (float)dims[1]); // x0, x1
        CLIP(oln[1], 0.0f, (float)dims[0]); CLIP(oln[3], 0.0f, (float)dims[0]); // y0, y1

        for (i = 4; i < (int)lsize; i++) oln[i] = 0.5f * (ln0[i] + ln1[i]);
    }

    free_array(darr);

    return M0;
}

static float find_overlap(float *img0, int *pts0, float *img1, int *pts1, float *data, const size_t *dims)
{
    /* Define overlap rectangle */
    /* Points are given by [j0, i0, j1, i1], where {j <-> y, i <-> x} */
    int pts[4] = {MIN(pts0[0], pts1[0]), MIN(pts0[1], pts1[1]), 
                  MAX(pts0[2], pts1[2]), MAX(pts0[3], pts1[3])};

    /* Image shapes are given by [Y, X],
       where Y = j1 - j0 is the number of rows and
             X = i1 - i0 is the number of columns
     */
    size_t s0[2] = {pts0[2] - pts0[0], pts0[3] - pts0[1]};
    size_t s1[2] = {pts1[2] - pts1[0], pts1[3] - pts1[1]};

    /* Calculate overlap and union */
    float ovl = 0.0f, unn = 0.0f;
    int i0, j0, i1, j1;
    float val0, val1;
    for (int j = pts[0]; j < pts[2]; j++)
    {
        for (int i = pts[1]; i < pts[3]; i++)
        {
            i0 = i - pts0[1]; j0 = j - pts0[0];
            if ((i0 >= 0) && (i0 < (int)s0[1]) && (j0 >= 0) && (j0 < (int)s0[0])) val0 = img0[j0 * s0[1] + i0];
            else val0 = 0.0f;

            i1 = i - pts1[1]; j1 = j - pts1[0];
            if ((i1 >= 0) && (i1 < (int)s1[1]) && (j1 >= 0) && (j1 < (int)s1[0])) val1 = img1[j1 * s1[1] + i1];
            else val1 = 0.0f;

            if (data[j * dims[1] + i] > 0.0)
            {
                ovl += data[j * dims[1] + i] * MIN(val0, val1);
                unn += data[j * dims[1] + i] * MAX(val0, val1);
            }
        }
    }

    if (unn) return ovl / unn;
    return 0.0f;
}

int filter_line(float *olines, unsigned char *proc, float *data, const size_t *dims, float *ilines,
                const size_t *ldims, float threshold, float dilation, line_profile profile)
{
    /* Check parameters */
    if (!olines|| !proc || !data || !ilines) {ERROR("filter_line: one of the arguments is NULL."); return -1;}
    if (!dims[0] && !dims[1]) {ERROR("filter_line: data array must have a positive size."); return -1;}

    if (ldims[0] == 0) return 0;
    if (*olines != *ilines) memcpy(olines, ilines, ldims[0] * ldims[1] * sizeof(float));

    int i, idx, pts[4], pt[2];
    int *ptp = pts;
    float I0, val;
    float *ln, *img;
    array darr = new_array(2, dims, sizeof(float), data);
    rect_iter ri;

    /* Filter faint lines */
    for (i = 0, ln = olines; i < (int)ldims[0]; i++, ln += ldims[1])
    {
        /* Check if processed already */
        if (proc[i] == 0) continue;

        /* Draw a line */
        create_line_image(&img, &ptp, dims, ln, dilation, profile);

        /* Calculate zero-th image moment */
        I0 = 0.0f;
        for (ri = ri_ini(2, pts, pts + 2); !ri_end(ri); ri_inc(ri))
        {
            pt[0] = ri->coord[0] + pts[0];
            pt[1] = ri->coord[1] + pts[1];
            RAVEL_INDEX(pt, &idx, darr);

            val = data[idx] * img[ri->index];
            I0 += (val < 0.0f) ? 0.0f : val;
        }
        ri_del(ri);

        /* Delete the line if M0 is too small */
        if (I0 < threshold)
        {
            memset(ln, 0, ldims[1] * sizeof(float));
            proc[i] = 0;
        }

        DEALLOC(img);
    }

    free_array(darr);

    return 0;
}

int group_line(float *olines, unsigned char *proc, float *data, const size_t *dims, float *ilines,
               const size_t *ldims, float cutoff, float threshold, float dilation, line_profile profile)
{
    /* Check parameters */
    if (!olines|| !data || !ilines) {ERROR("group_line: one of the arguments is NULL."); return -1;}
    if (!dims[0] && !dims[1]) {ERROR("group_line: data array must have a positive size."); return -1;}
    
    if (ldims[0] == 0) return 0;
    if (*olines != *ilines) memcpy(olines, ilines, ldims[0] * ldims[1] * sizeof(float));

    int i, j, counter;
    float corr, M0;
    float *ln0, *ln1;
    // Coordinates and indices for the tree
    float *coords = MALLOC(float, 2 * ldims[0]);
    int *idxs = MALLOC(int, ldims[0]);
    // Mask of the lines which have been already used
    unsigned char *used = calloc(ldims[0], sizeof(unsigned char));

    for (i = 0, ln0 = olines; i < (int)ldims[0]; i++, ln0 += ldims[1])
    {
        coords[2 * i] = 0.5f * (ln0[0] + ln0[2]);
        coords[2 * i + 1] = 0.5f * (ln0[1] + ln0[3]);
        idxs[i] = i;
    }

    kd_tree tree = kd_build(coords, ldims[0], 2, idxs, sizeof(int));
    float *img_pair, *oimg;
    int pts_pair[4], opts[4];
    int *ptp_pair = pts_pair, *optp = opts;
    float *oln = MALLOC(float, ldims[1]);

    do
    {
        counter = 0;
        for (i = 0, ln0 = olines; i < (int)ldims[0]; i++, ln0 += ldims[1])
        {
            /* check if processed already */
            if (!proc[i]) continue;

            /* check if the line should be updated */
            if (!used[i])
            {
                used[i] = 1;
                /* find all the lines in a range */
                query_stack stack = kd_find_range(tree, coords + 2 * i, cutoff);

                for (query_stack node = stack; node; node = node->next)
                {
                    j = *(int *)node->query->node->data;
                    ln1 = olines + j * ldims[1];

                    if (proc[j] && i != j)
                    {
                        /* Create an image of a pair */
                        create_line_image_pair(&img_pair, &ptp_pair, dims, ln0, ln1, dilation, profile);
                        
                        /* Collapse a pair of lines */
                        M0 = collapse_pair(oln, img_pair, pts_pair, data, dims, ln0, ln1, ldims[1]);

                        if (M0 > 0.0f)
                        {
                            /* Create an image of oln */
                            create_line_image(&oimg, &optp, dims, oln, dilation, profile);

                            /* Find overlap between imp_pair and imp_oln */
                            corr = find_overlap(img_pair, pts_pair, oimg, opts, data, dims);

                            if (corr > threshold)
                            {
                                memcpy(ln0, oln, ldims[1] * sizeof(float));
                                memset(ln1, 0, ldims[1] * sizeof(float));
                                kd_delete(tree, node->query->node->pos);
                                proc[j] = 0; used[i] = 0; counter++;

                                break;
                            }

                            DEALLOC(oimg);
                        }

                        DEALLOC(img_pair);
                    }
                }

                free_stack(stack);
            }
        }
    } while (counter);

    kd_free(tree); DEALLOC(coords); DEALLOC(idxs); DEALLOC(used); DEALLOC(oln);

    return 0;
}

int normalise_line(float *out, float *data, const size_t *dims, float *lines,
                   const size_t *ldims, float dilations[3], line_profile profile)
{
    /* Check parameters */
    if (!out || !data || !lines) {ERROR("normalise_line: one of the arguments is NULL."); return -1;}
    if (!dims[0] && !dims[1]) {ERROR("normalise_line: data array must have a positive size."); return -1;}

    if (ldims[0] == 0) return 0;

    int i, idx, len, pts[4], pt[2];
    int *ptp = pts;
    float bgd, div, val;
    float *ln, *buffer, *img;
    rect_iter ri;

    array marr = new_array(2, dims, sizeof(unsigned), calloc(dims[0] * dims[1], sizeof(unsigned)));
    array Iarr = new_array(2, dims, sizeof(float), calloc(dims[0] * dims[1], sizeof(float)));
    array Warr = new_array(2, dims, sizeof(int), calloc(dims[0] * dims[1], sizeof(int)));

    for (i = 0, ln = lines; i < (int)ldims[0]; i++, ln += ldims[1])
    {
        /* Convert coordinates to points */
        pts[0] = roundf(ln[1]); pts[1] = roundf(ln[0]);
        pts[2] = roundf(ln[3]); pts[3] = roundf(ln[2]);
        plot_line_width(marr, marr->dims, pts, ln[4] + dilations[1], 1,
                        set_pixel_int, tophat_profile);
    }

    for (i = 0, ln = lines; i < (int)ldims[0]; i++, ln += ldims[1])
    {
        /* Calculate bgd = median((img - streak_mask) * data) */
        create_line_image(&img, &ptp, dims, ln, dilations[2], tophat_profile);
        buffer = MALLOC(float, (pts[2] - pts[0]) * (pts[3] - pts[1]));
        len = 0;

        for (ri = ri_ini(2, pts, pts + 2); !ri_end(ri); ri_inc(ri))
        {
            pt[0] = ri->coord[0] + pts[0];
            pt[1] = ri->coord[1] + pts[1];

            RAVEL_INDEX(pt, &idx, Iarr);

            if (img[ri->index] > 0.0f && !GET(marr, unsigned, idx))
            {
                buffer[len++] = data[idx];
            }
        }
        DEALLOC(img); ri_del(ri);

        bgd = *(float *)wirthmedian(buffer, len, sizeof(float), compare_float);

        DEALLOC(buffer);

        /* Calculate div = max(img * data) */
        create_line_image(&img, &ptp, dims, ln, dilations[0], profile);
        div = 0.0f;
        for (ri = ri_ini(2, pts, pts + 2); !ri_end(ri); ri_inc(ri))
        {
            pt[0] = ri->coord[0] + pts[0];
            pt[1] = ri->coord[1] + pts[1];
            RAVEL_INDEX(pt, &idx, Iarr);
            val = data[idx] * img[ri->index];
            if (div < val) div = val;
        }
        ri_del(ri);

        /* Write down a normalised reflection profile
           p = (data * img - bgd) / (div - bgd)
         */
        div = 1.0f / (div - bgd);
        for (ri = ri_ini(2, pts, pts + 2); !ri_end(ri); ri_inc(ri))
        {
            pt[0] = ri->coord[0] + pts[0];
            pt[1] = ri->coord[1] + pts[1];
            RAVEL_INDEX(pt, &idx, Iarr);

            if (img[ri->index])
            {
                val = data[idx] * img[ri->index];
                if (val > bgd) GET(Iarr, float, idx) += div * (val - bgd);
                GET(Warr, int, idx) += 1;
            }
        }
        DEALLOC(img); ri_del(ri);
    }

    DEALLOC(marr->data); free_array(marr);

    for (i = 0; i < (int)(dims[0] * dims[1]); i++)
    {
        out[i] = GET(Warr, int, i) ? GET(Iarr, float, i) / GET(Warr, int, i) : 0.0f;
    }

    DEALLOC(Iarr->data); free_array(Iarr);
    DEALLOC(Warr->data); free_array(Warr);

    return 0;
}

int refine_line(float *olines, float *data, const size_t *dims, float *ilines, const size_t *ldims,
                float dilation, line_profile profile)
{
    if (!olines || !data || !ilines) {ERROR("refine_line: one of the arguments is NULL."); return -1;}
    if (!dims[0] && !dims[1]) {ERROR("refine_line: data array must have a positive size."); return -1;}

    if (ldims[0] == 0) return 0;
    if (*olines != *ilines) memcpy(olines, ilines, ldims[0] * ldims[1] * sizeof(float));

    int i, idx, pts[4], pt[2];
    int *ptp = pts;
    float M0, MX, MXX, mu_x, mu_xx, x, val, tmax, tau[2];
    float *ln, *img;

    /* Create data array */
    array darr = new_array(2, dims, sizeof(float), data);
    rect_iter ri;

    for (i = 0, ln = olines; i < (int)ldims[0]; i++, ln += ldims[1])
    {
        create_line_image(&img, &ptp, dims, ln, dilation, profile);

        tmax = sqrtf(SQ(ln[2] - ln[0]) + SQ(ln[3] - ln[1]));
        tau[0] = (ln[2] - ln[0]) / tmax; tau[1] = (ln[3] - ln[1]) / tmax;

        /* Calculate image moments */
        M0 = MX = MXX = 0.0f;
        for (ri = ri_ini(2, pts, pts + 2); !ri_end(ri); ri_inc(ri))
        {
            /* point is given by [j, i], where {j <-> y, i <-> x} */
            pt[0] = ri->coord[0] + pts[0];
            pt[1] = ri->coord[1] + pts[1];
            RAVEL_INDEX(pt, &idx, darr);

            val = data[idx] * img[ri->index];
            val = (val < 0.0f) ? 0.0f : val;
            x = (pt[0] - ln[1]) * tau[0] - (pt[1] - ln[0]) * tau[1];
            M0 += val; MX += x * val; MXX += SQ(x) * val;
        }
        DEALLOC(img); ri_del(ri);

        if (M0 > 0.0f)
        {
            /* Central moments */
            mu_x = MX / M0; mu_xx = MXX / M0 - mu_x * mu_x;

            /* Shift the line */
            ln[0] -= tau[1] * mu_x; ln[1] += tau[0] * mu_x;
            ln[2] -= tau[1] * mu_x; ln[3] += tau[0] * mu_x;
            ln[4] = sqrtf(8.0f * mu_xx);
        }
    }

    free_array(darr);

    return 0;
}

/*----------------------------------------------------------------------------*/
/*------------------------------- Euler angles -------------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/*  Euler angles with Bunge convention

        ang   =  [phi1, Phi, phi2]
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

        ang    =  [theta, alpha, beta]
        theta \el [0, 2 * M_PI)         Angle of rotation
        alpha \el [0, M_PI)             Angle between the axis of rotation and 0Z
        phi   \el [0, 2 * M_PI)         Polar angle of the axis of rotation
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

/*---------------------------------------------------------------------------
                        Model refinement criterion
---------------------------------------------------------------------------*/

double cross_entropy(unsigned *ij, float *p, unsigned *fidxs, size_t *dims, float **lines, const size_t *ldims,
                     size_t lsize, float dilation, float epsilon, line_profile profile, unsigned threads)
{
    if (!ij || !p || !dims || !lines || !ldims) {ERROR("cross_entropy: one of the arguments is NULL."); return 0.0;}
    if (lsize == 0 || dims[0] == 0 || dims[1] == 0) return 0.0;

    double entropy = 0.0;
    threads = (threads > lsize) ? lsize : threads;

    #pragma omp parallel num_threads(threads)
    {
        int j, ln_area, k0, k, pts[4];
        float wd, tmax, val, t, tau[2];
        float *ln;
        double crit;

        index_tuple idxs;
        array iarr = new_array(2, dims, sizeof(float), MALLOC(float, dims[0] * dims[1]));

        #pragma omp for
        for (int i = 0; i < (int)lsize; i++)
        {
            idxs = new_index_tuple(1);
            memset(iarr->data, 0, iarr->size * iarr->item_size);

            for (j = 0, ln = lines[i]; j < (int)ldims[2 * i]; j++, ln += ldims[2 * i + 1])
            {
                /* Convert coordinates to points */
                pts[0] = roundf(ln[1]); pts[1] = roundf(ln[0]); pts[2] = roundf(ln[3]); pts[3] = roundf(ln[2]);

                wd = ln[4] + dilation;
                k0 = idxs->size;
                ln_area = (2 * ((int)wd + 1.0f) + abs(pts[2] - pts[0])) * (2 * ((int)wd + 1.0f) + abs(pts[3] - pts[1]));

                /* Expand the list if needed */
                if (idxs->max_size < idxs->size + (size_t)ln_area)
                    realloc_index_tuple(idxs, idxs->size + (size_t)ln_area);

                /* Write down the indices */
                plot_line_width(idxs, dims, pts, wd + 1.0f, 1, set_pixel_index, profile);

                /* Find the unit vector */
                tmax = sqrtf(SQ(ln[2] - ln[0]) + SQ(ln[3] - ln[1]));
                tau[0] = (ln[2] - ln[0]) / tmax; tau[1] = (ln[3] - ln[1]) / tmax;
                wd = 0.5f * (wd + 1.0f);

                for (k = k0; k < (int)idxs->size; k++)
                {
                    if (idxs->val[k] > 0.0f)
                    {
                        /* Find distance to the line */
                        val = SQ((idxs->y[k] - ln[1]) * tau[0] - (idxs->x[k] - ln[0]) * tau[1]);
                        t = (idxs->x[k] - ln[0]) * tau[0] + (idxs->y[k] - ln[1]) * tau[1];
                        if (t < 0.0) val += SQ(t);
                        if (t > tmax) val += SQ(t - tmax);

                        val = profile(sqrtf(val), wd);
                        set_pixel_float(iarr, idxs->x[k], idxs->y[k], val);
                    }
                }
            }

            free_index_tuple(idxs);

            /* Calculate the cross entropy: ce = -p * log(q) */
            crit = 0.0;
            for (j = fidxs[i]; j < (int)fidxs[i + 1]; j++)
            {
                crit -= p[j] * log(MAX(GET(iarr, float, ij[j]), epsilon)); 
            }

            #pragma omp atomic
            entropy += crit;
        }

        DEALLOC(iarr->data); free_array(iarr);
    }

    return entropy;
}