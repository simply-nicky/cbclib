#include "img_proc.h"
#include "array.h"

/** pi / 9 **/
#define VAR_ANG_MIN 0.3490658503988658956
/** pi / 3 **/
#define VAR_ANG_MAX 1.047197551196597853
#define TOL 3.1425926535897937e-05

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
    free((void *) n_tuple->values);
    free((void *) n_tuple);
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
    n_tuple->values = (unsigned int *)realloc((void *)n_tuple->values, n_tuple->dim * n_tuple->max_size * sizeof(unsigned int));
    if (n_tuple->values == NULL) ERROR("not enough memory.");
}

static void add_4tuple(ntuple_list list, double v1, double v2, double v3, double v4)
{
    /* check parameters */
    if (list == NULL) ERROR("add_4tuple: invalid n-tuple input.");
    if (list->dim != 4) ERROR("add_4tuple: the n-tuple must be a 7-tuple.");

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

#define CLIP(_coord, _min, _max)                \
{                                               \
    _coord = (_coord > _min) ? _coord : _min;   \
    _coord = (_coord < _max) ? _coord : _max;   \
}

static void plot_line_width(void *out, size_t *dims, rect ln, double wd, unsigned int max_val, set_pixel setter)
{
    /* plot an anti-aliased line of width wd */
    int dx = abs(ln->x1 - ln->x0), sx = ln->x0 < ln->x1 ? 1 : -1;
    int dy = abs(ln->y1 - ln->y0), sy = ln->y0 < ln->y1 ? 1 : -1;
    int err = dx - dy, derr = 0, dx0 = 0, e2, x2, y2, val;    /* error value e_xy */
    double ed = dx + dy == 0 ? 1 : sqrt((double)dx * dx + (double)dy * dy);
    wd = (wd + 1) / 2;

    /* define line bounds */
    int wi = wd;
    rect bnd = (rect)malloc(sizeof(struct rect_s));

    if (ln->x0 < ln->x1)
    {
        bnd->x0 = ln->x0 - wi; CLIP(bnd->x0, 0, (int)dims[1] - 1);
        bnd->x1 = ln->x1 + wi; CLIP(bnd->x1, 0, (int)dims[1] - 1);
        err += (ln->x0 - bnd->x0) * dy; ln->x0 = bnd->x0; ln->x1 = bnd->x1;
    }
    else
    {
        bnd->x0 = ln->x1 - wi; CLIP(bnd->x0, 0, (int)dims[1] - 1);
        bnd->x1 = ln->x0 + wi; CLIP(bnd->x1, 0, (int)dims[1] - 1);
        err += (bnd->x1 - ln->x0) * dy; ln->x0 = bnd->x1; ln->x1 = bnd->x0;
    }
    if (ln->y0 < ln->y1)
    {
        bnd->y0 = ln->y0 - wi; CLIP(bnd->y0, 0, (int)dims[0] - 1);
        bnd->y1 = ln->y1 + wi; CLIP(bnd->y1, 0, (int)dims[0] - 1);
        err -= (ln->y0 - bnd->y0) * dx; ln->y0 = bnd->y0; ln->y1 = bnd->y1;
    }
    else
    {
        bnd->y0 = ln->y1 - wi; CLIP(bnd->y0, 0, (int)dims[0] - 1);
        bnd->y1 = ln->y0 + wi; CLIP(bnd->y1, 0, (int)dims[0] - 1);
        err -= (bnd->y1 - ln->y0) * dx; ln->y0 = bnd->y1; ln->y1 = bnd->y0;
    }

    while (1)
    {
        /* pixel loop */
        err += derr; derr = 0;
        ln->x0 += dx0; dx0 = 0;
        val = max_val - fmax(max_val * (abs(err - dx + dy) / ed - wd + 1), 0.0);
        setter(out, ln->x0, ln->y0, val, max_val);

        if (2 * err >= -dx)
        {
            /* x step */
            for (e2 = err + dy, y2 = ln->y0 + sy;
                 abs(e2) < ed * wd && y2 >= bnd->y0 && y2 <= bnd->y1;
                 e2 += dx, y2 += sy)
            {
                val = max_val - fmax(max_val * (abs(e2) / ed - wd + 1), 0.0);
                setter(out, ln->x0, y2, val, max_val);
            }
            if (ln->x0 == ln->x1) break;
            derr -= dy; dx0 += sx;
        }
        if (2 * err <= dy)
        {
            /* y step */
            for (e2 = err - dx, x2 = ln->x0 + sx;
                 abs(e2) < ed * wd && x2 >= bnd->x0 && x2 <= bnd->x1;
                 e2 -= dy, x2 += sx)
            {
                val = max_val - fmax(max_val * (abs(e2) / ed - wd + 1), 0.0);
                setter(out, x2, ln->y0, val, max_val);
            }
            if (ln->y0 == ln->y1) break;
            derr += dx; ln->y0 += sy;
        }
    }

    free(bnd);
}

int draw_lines(unsigned int *out, size_t Y, size_t X, unsigned int max_val, double *lines, size_t n_lines, unsigned int dilation)
{
    /* check parameters */
    if (!out || !lines) {ERROR("draw_lines: one of the arguments is NULL."); return -1;}
    if (!X && !Y) {ERROR("draw_lines: image size must be positive."); return -1;}

    if (n_lines == 0) return 0;

    size_t ldims[2] = {n_lines, 7};
    size_t odims[2] = {Y, X};
    array larr = new_array(2, ldims, sizeof(double), lines);
    array oarr = new_array(2, odims, sizeof(unsigned int), out);

    line ln = init_line(larr, 1);
    rect rt = (rect)malloc(sizeof(struct rect_s));
    double *ln_ptr;

    for (int i = 0; i < (int)n_lines; i++)
    {
        UPDATE_LINE(ln, i);
        ln_ptr = ln->data;

        if (ln_ptr[0] > 0.0 && ln_ptr[0] < (double)X &&
            ln_ptr[1] > 0.0 && ln_ptr[1] < (double)Y &&
            ln_ptr[2] > 0.0 && ln_ptr[2] < (double)X &&
            ln_ptr[3] > 0.0 && ln_ptr[3] < (double)Y)
        {
            rt->x0 = round(ln_ptr[0]); rt->y0 = round(ln_ptr[1]);
            rt->x1 = round(ln_ptr[2]); rt->y1 = round(ln_ptr[3]);

            plot_line_width((void *)oarr, oarr->dims, rt, ln_ptr[4] + (double)dilation, max_val, set_pixel_color);
        }
    }

    free(ln); free(rt);
    free_array(larr); free_array(oarr);

    return 0;
}

int draw_line_indices(unsigned int **out, size_t *n_idxs, size_t Y, size_t X, unsigned int max_val, double *lines, size_t n_lines, unsigned int dilation)
{
    /* check parameters */
    if (!lines) {ERROR("draw_line_indices: lines is NULL."); return -1;}
    if (!X && !Y) {ERROR("draw_line_indices: image size must be positive."); return -1;}

    if (n_lines == 0) return 0;

    size_t ldims[2] = {n_lines, 7};
    size_t odims[2] = {Y, X};
    array larr = new_array(2, ldims, sizeof(double), lines);
    ntuple_list idxs = new_ntuple_list(4, 1);

    line ln = init_line(larr, 1);
    rect rt = (rect)malloc(sizeof(struct rect_s));
    double *ln_ptr;
    double wd;
    int ln_area;

    for (idxs->iter = 0; idxs->iter < (int)n_lines; idxs->iter++)
    {
        UPDATE_LINE(ln, idxs->iter);
        ln_ptr = ln->data;

        if (ln_ptr[0] > 0.0 && ln_ptr[0] < (double)X &&
            ln_ptr[1] > 0.0 && ln_ptr[1] < (double)Y &&
            ln_ptr[2] > 0.0 && ln_ptr[2] < (double)X &&
            ln_ptr[3] > 0.0 && ln_ptr[3] < (double)Y)
        {
            rt->x0 = round(ln_ptr[0]); rt->y0 = round(ln_ptr[1]);
            rt->x1 = round(ln_ptr[2]); rt->y1 = round(ln_ptr[3]);
            wd = ln_ptr[4] + (double)dilation;
            ln_area = (2 * (int)wd + abs(rt->x1 - rt->x0)) * (2 * (int)wd + abs(rt->y1 - rt->y0));

            if (idxs->max_size < idxs->size + (size_t)ln_area)
                realloc_ntuple_list(idxs, idxs->size + (size_t)ln_area);

            plot_line_width((void *)idxs, odims, rt, wd, max_val, set_pixel_index);
        }
    }

    realloc_ntuple_list(idxs, idxs->size);
    *out = idxs->values;
    *n_idxs = idxs->size;

    free(ln); free(rt);
    free_array(larr); free(idxs);

    return 0;
}

static void draw_pair(unsigned int **out, rect orect, size_t Y, size_t X,
    unsigned int max_val, double *ln0, double *ln1, unsigned int dilation)
{
    /* Define the bounds 'rt0' of the former line 'ln0' */
    rect rt0 = (rect)malloc(sizeof(struct rect_s));
    rt0->x0 = round(ln0[0]); rt0->y0 = round(ln0[1]);
    rt0->x1 = round(ln0[2]); rt0->y1 = round(ln0[3]);

    /* Define the bounds 'rt1' of the latter line 'ln1' */
    rect rt1 = (rect)malloc(sizeof(struct rect_s));
    rt1->x0 = round(ln1[0]); rt1->y0 = round(ln1[1]);
    rt1->x1 = round(ln1[2]); rt1->y1 = round(ln1[3]);

    int wi = (ln0[4] > ln1[4]) ? ln0[4] : ln1[4];

    /* Find the outer bounds 'orect' */
    if (rt0->x0 < rt0->x1) {orect->x0 = rt0->x0; orect->x1 = rt0->x1; }
    else {orect->x0 = rt0->x1; orect->x1 = rt0->x0; }
    if (rt1->x0 < orect->x0) orect->x0 = rt1->x0;
    if (rt1->x1 < orect->x0) orect->x0 = rt1->x1;
    if (rt1->x0 > orect->x1) orect->x1 = rt1->x0;
    if (rt1->x1 > orect->x1) orect->x1 = rt1->x1;

    if (rt0->y0 < rt0->y1) {orect->y0 = rt0->y0; orect->y1 = rt0->y1; }
    else {orect->y0 = rt0->y1; orect->y1 = rt0->y0; }
    if (rt1->y0 < orect->y0) orect->y0 = rt1->y0;
    if (rt1->y1 < orect->y0) orect->y0 = rt1->y1;
    if (rt1->y0 > orect->y1) orect->y1 = rt1->y0;
    if (rt1->y1 > orect->y1) orect->y1 = rt1->y1;

    /* Expand the bounds 'orect' by the line's width */
    orect->x0 = (orect->x0 - wi > 0) ? orect->x0 - wi : 0;
    orect->y0 = (orect->y0 - wi > 0) ? orect->y0 - wi : 0;
    orect->x1 = (orect->x1 + wi < (int)X) ? orect->x1 + wi : (int)X;
    orect->y1 = (orect->y1 + wi < (int)Y) ? orect->y1 + wi : (int)Y;

    /* Create an image 'oarr' */
    size_t odims[2] = {orect->y1 - orect->y0, orect->x1 - orect->x0};
    (*out) = (unsigned int *)calloc(odims[0] * odims[1], sizeof(unsigned int));
    array oarr = new_array(2, odims, sizeof(unsigned int), *out);

    rt0->x0 -= orect->x0; rt0->y0 -= orect->y0; rt0->x1 -= orect->x0; rt0->y1 -= orect->y0;
    rt1->x0 -= orect->x0; rt1->y0 -= orect->y0; rt1->x1 -= orect->x0; rt1->y1 -= orect->y0;

    /* Plot the lines */
    plot_line_width((void *)oarr, oarr->dims, rt0, ln0[4] + (double)dilation, max_val, set_pixel_color);
    plot_line_width((void *)oarr, oarr->dims, rt1, ln1[4] + (double)dilation, max_val, set_pixel_color);

    free(rt0); free(rt1); free_array(oarr);
}

static void collapse_pair(unsigned int *img, rect img_rt, double *data, size_t Y, size_t X,
    double *ln0, double *ln1, size_t lsize)
{
    int j, k;
    unsigned int *img_j, *img_jk;
    double MX, MY, MXY, MXX, MYY, M0;
    double mu_x, mu_y, mu_xy, mu_xx, mu_yy, th, w, val, t0, t1;
    double *data_j, *data_jk;

    /* Image moments */
    M0 = MX = MY = MXY = MXX = MYY = 0.0;
    for (j = img_rt->y0, img_j = img, data_j = data + X * img_rt->y0 + img_rt->x0;
         j < img_rt->y1; j++, img_j += img_rt->x1 - img_rt->x0, data_j += X)
        for (k = img_rt->x0, img_jk = img_j, data_jk = data_j;
             k < img_rt->x1; k++, img_jk++, data_jk++)
        {
            val = (*data_jk) * (double)(*img_jk);
            M0 += val; MX += k * val; MY += j * val;
            MXY += k * j * val; MYY += j * j * val; MXX += k * k * val;
        }

    if (M0)
    {
        /* Central moments */
        mu_x = MX / M0; mu_y = MY / M0;
        mu_xy = 2.0 * (MXY / M0 - mu_x * mu_y);
        mu_xx = MXX / M0 - mu_x * mu_x; mu_yy = MYY / M0 - mu_y * mu_y;

        /* Orientation and major axis length */ 
        th = 0.5 * atan(mu_xy / (mu_xx - mu_yy));
        if (mu_xx < mu_yy) th += M_PI_2;
        w = 0.5 * sqrt(8.0 * (mu_xx + mu_yy - sqrt(SQ(mu_xy) + SQ(mu_xx - mu_yy))));

        /* Collapse the lines */
        t0 = (ln0[0] - mu_x) * cos(th) + (ln0[1] - mu_y) * sin(th);
        t1 = (ln1[0] - mu_x) * cos(th) + (ln1[1] - mu_y) * sin(th);

        if (fabs(t0) > fabs(t1))
        {
            ln0[0] = mu_x + t0 * cos(th);       // x0
            ln0[1] = mu_y + t0 * sin(th);       // y0
        }
        else
        {
            ln0[0] = mu_x + t1 * cos(th);       // x0
            ln0[1] = mu_y + t1 * sin(th);       // y0
        }

        t0 = (ln0[2] - mu_x) * cos(th) + (ln0[3] - mu_y) * sin(th);
        t1 = (ln1[2] - mu_x) * cos(th) + (ln1[3] - mu_y) * sin(th);

        if (fabs(t0) > fabs(t1))
        {
            ln0[2] = mu_x + t0 * cos(th);       // x1
            ln0[3] = mu_y + t0 * sin(th);       // y1
        }
        else
        {
            ln0[2] = mu_x + t1 * cos(th);       // x1
            ln0[3] = mu_y + t1 * sin(th);       // y1
        }
        ln0[4] = w;                             // width

        CLIP(ln0[0], 0.0, (double)X); CLIP(ln0[2], 0.0, (double)X);
        CLIP(ln0[1], 0.0, (double)Y); CLIP(ln0[3], 0.0, (double)Y);

        memset(ln1, 0, lsize * sizeof(double));
    }
}

#define WRAP_DIST(_dist, _dx, _hb, _fb, _div)   \
{                                               \
    double _dx1;                                \
    if (_dx < -_hb) _dx1 = _dx + _fb;           \
    else if (_dx > _hb) _dx1 = _dx - _fb;       \
    else _dx1 = _dx;                            \
    _dist += SQ(_dx1 / _div);                   \
}

static int indirect_cmp(const void *a, const void *b, void *data)
{
    double *dptr = data;
    if (dptr[*(size_t *)a] > dptr[*(size_t *)b]) return 1;
    else if (dptr[*(size_t *)a] < dptr[*(size_t *)b]) return -1;
    else return 0;
}

int filter_lines(double *olines, double *data, size_t Y, size_t X, double *ilines, size_t n_lines,
    double x_c, double y_c, double radius)
{
    /* Check parameters */
    if (!olines|| !data || !ilines) {ERROR("filter_lines: one of the arguments is NULL."); return -1;}
    if (!X && !Y) {ERROR("filter_lines: data array must have a positive size."); return -1;}
    if (x_c <= 0 || x_c > X || y_c <= 0 || y_c > Y)
    {ERROR("filter_lines: center coordinates are out of bounds."); return -1;}
    if (radius <= 0.0) {ERROR("filter_lines: radius must be positive."); return -1;}
    
    if (n_lines == 0) return 0;

    size_t ldims[2] = {n_lines, 7};
    array ilarr = new_array(2, ldims, sizeof(double), ilines);
    array olarr = new_array(2, ldims, sizeof(double), olines);

    line iln = init_line(ilarr, 1);
    line oln = init_line(olarr, 1);

    double max_wd = 0.0, max_l = 0.0, mean_r = 0.0, var_a = 0.0;
    double mean_a = 0.0, mean_a2 = 0.0, min_d, d, dth, *oln_ptr;

    /* Central coordinates */
    double *als = (double *)malloc(n_lines * sizeof(double));   // angle between the line and the central tangent
    double *ls = (double *)malloc(n_lines * sizeof(double));    // line's length
    double *rs = (double *)malloc(n_lines * sizeof(double));    // line's radius
    double *ths = (double *)malloc(n_lines * sizeof(double));   // line's polar angle
    int i, j, min_idx, n_pairs = 0;

    /* Get the coordinates and copy ilines to olines */
    for (i = 0; i < (int)n_lines; i++)
    {
        UPDATE_LINE(iln, i);
        UPDATE_LINE(oln, i);

        if (*olines != *ilines) memcpy(oln->data, iln->data, iln->line_size);
        oln_ptr = oln->data;

        /* Swap the line ends, so all the lines are clockwise aligned */
        dth = atan2(oln_ptr[1] - y_c, oln_ptr[0] - x_c) - atan2(oln_ptr[3] - y_c, oln_ptr[2] - x_c);
        if (dth < - M_PI) dth += 2 * M_PI;
        if (dth > M_PI) dth -= 2 * M_PI;

        if (dth > 0)
        {
            SWAP(oln_ptr[0], oln_ptr[2], double);
            SWAP(oln_ptr[1], oln_ptr[3], double);
        }
        if (oln_ptr[4] > max_wd) max_wd = oln_ptr[4];

        /* Central coordinates */
        rs[i] = sqrt(SQ((oln_ptr[0] + oln_ptr[2]) / 2 - x_c) + SQ((oln_ptr[1] + oln_ptr[3]) / 2 - y_c));
        ths[i] = atan2((oln_ptr[1] + oln_ptr[3]) / 2 - y_c, (oln_ptr[0] + oln_ptr[2]) / 2 - x_c);
        als[i] = fabs(atan2(oln_ptr[3] - oln_ptr[1], oln_ptr[2] - oln_ptr[0]) - ths[i] + M_PI_2) - M_PI;
        ls[i] = sqrt(SQ(oln_ptr[3] - oln_ptr[1]) + SQ(oln_ptr[2] - oln_ptr[0]));

        /* Calculate als variance */
        mean_a2 += (als[i] - mean_a2) / (i + 1);
        var_a += (als[i] - mean_a) * (als[i] - mean_a2);
        mean_a = mean_a2;

        /* Calculate rs mean and ls max */
        mean_r += (rs[i] - mean_r) / (i + 1);
        if(ls[i] > max_l) max_l = ls[i];
    }

    free(iln); free_array(ilarr);

    var_a = sqrt(var_a / n_lines);
    CLIP(var_a, VAR_ANG_MIN, VAR_ANG_MAX);
    max_l = max_l / mean_r;

    unsigned int *pairs = (unsigned int *)malloc(2 * n_lines * sizeof(unsigned int));
    double *ds = (double *)malloc(n_lines * sizeof(double));

    /* Find the closest line for each line and filter out the bad lines */
    for (i = 0; i < (int)n_lines; i++)
    {
        UPDATE_LINE(oln, i);

        if (fabs(als[i]) < var_a + ls[i] / rs[i])
        {
            min_d = DBL_MAX; min_idx = -1;
            for (j = 0; j < (int)n_lines; j++)
            {
                if (fabs(als[j]) < var_a + ls[j] / rs[j] && j != i)
                {
                    d = SQ(0.5 * (rs[j] - rs[i]) / max_wd);
                    WRAP_DIST(d, ths[j] - ths[i], M_PI, 2 * M_PI, max_l);
                    WRAP_DIST(d, als[j] - als[i], M_PI_2, M_PI, var_a);

                    if (d < SQ(radius) && d < min_d) {min_d = d; min_idx = j;}
                }
            }
            if (min_idx >= 0)
            {
                pairs[2 * n_pairs] = i;
                pairs[2 * n_pairs + 1] = min_idx;
                ds[n_pairs] = min_d; n_pairs++;
            }
        }
        else memset(oln->data, 0, oln->line_size);
    }

    free(oln); free(als); free(ls); free(rs); free(ths);

    size_t *inds = (size_t *)malloc(n_pairs * sizeof(size_t));
    for (i = 0; i < n_pairs; i++) inds[i] = i;

    /* Sort the pairs based on the distance */
    qsort_r(inds, n_pairs, sizeof(size_t), indirect_cmp, ds);
    free(ds);

    unsigned int *img;
    rect orect = (rect)malloc(sizeof(struct rect_s));
    double *ln0, *ln1;

    /* Collapse the pairs */
    for (i = 0; i < n_pairs; i++)
    {
        ln0 = olines + ldims[1] * pairs[2 * inds[i]];
        ln1 = olines + ldims[1] * pairs[2 * inds[i] + 1];

        if (ln0[0] && ln1[0])
        {
            /* Create an image of a pair */
            draw_pair(&img, orect, Y, X, 255, ln0, ln1, 0);
            
            /* Collapse a pair of lines */
            collapse_pair(img, orect, data, Y, X, ln0, ln1, ldims[1]);

            free(img);
        }
    }

    free(inds); free(pairs); free(orect); free_array(olarr);

    return 0;
}

static void rotmat_to_euler(double *eul, double *rm)
{
    eul[1] = acos(rm[8]);
    if (fabs(eul[1]) < 1e-8) {eul[0] = atan2(-rm[3], rm[0]); eul[2] = 0.0; }
    else if (fabs(M_PI - eul[1]) < TOL)
    {eul[0] = atan2(rm[3], rm[0]); eul[2] = 0.0; }
    else {eul[0] = atan2(rm[6], -rm[7]); eul[2] = atan2(rm[2], rm[5]); }
}

int compute_euler_angles(double *eulers, double *rot_mats, size_t n_mats)
{
    /* check parameters */
    if (!eulers || !rot_mats) {ERROR("compute_euler_angles: one of the arguments is NULL."); return -1;}
    if (!n_mats) {ERROR("compute_euler_angles: number of matrices must be positive."); return -1;}   

    size_t rm_dims[2] = {n_mats, 9};
    array rm_arr = new_array(2, rm_dims, sizeof(double), rot_mats);
    line rm_ln = init_line(rm_arr, 1);

    size_t e_dims[2] = {n_mats, 3};
    array e_arr = new_array(2, e_dims, sizeof(double), eulers);
    line e_ln = init_line(e_arr, 1);

    for (int i = 0; i < (int)n_mats; i++)
    {
        UPDATE_LINE(rm_ln, i);
        UPDATE_LINE(e_ln, i);

        rotmat_to_euler(e_ln->data, rm_ln->data);
    }

    free(e_ln); free(rm_ln);
    free_array(e_arr); free_array(rm_arr);

    return 0;
}

static void euler_to_rotmat(double *rm, double *eul)
{
    double c0 = cos(eul[0]), c1 = cos(eul[1]), c2 = cos(eul[2]);
    double s0 = sin(eul[0]), s1 = sin(eul[1]), s2 = sin(eul[2]);

    rm[0] = c0 * c1 - s0 * s1 * c2;
    rm[1] = s0 * c1 + c0 * s1 * c2;
    rm[2] = s1 * s2;
    rm[3] = -c0 * s1 - s0 * c1 * c2;
    rm[4] = -s0 * s1 + c0 * c1 * c2;
    rm[5] = c1 * s2;
    rm[6] = s0 * s2;
    rm[7] = -c0 * s2;
    rm[8] = c2;
}

int compute_rot_matrix(double *rot_mats, double *eulers, size_t n_mats)
{
    /* check parameters */
    if (!eulers || !rot_mats) {ERROR("compute_rot_matrix: one of the arguments is NULL."); return -1;}
    if (!n_mats) {ERROR("compute_rot_matrix: number of matrices must be positive."); return -1;}   

    size_t rm_dims[2] = {n_mats, 9};
    array rm_arr = new_array(2, rm_dims, sizeof(double), rot_mats);
    line rm_ln = init_line(rm_arr, 1);

    size_t e_dims[2] = {n_mats, 3};
    array e_arr = new_array(2, e_dims, sizeof(double), eulers);
    line e_ln = init_line(e_arr, 1);

    for (int i = 0; i < (int)n_mats; i++)
    {
        UPDATE_LINE(rm_ln, i);
        UPDATE_LINE(e_ln, i);

        euler_to_rotmat(rm_ln->data, e_ln->data);
    }

    free(e_ln); free(rm_ln);
    free_array(e_arr); free_array(rm_arr);

    return 0;
}

static void calc_rotmat(double *rm, double th, double bb, double cc, double dd)
{
    double a = cos(th), b = bb * sin(th), c = cc * sin(th), d = dd * sin(th);

    rm[0] = a * a + b * b - c * c - d * d;
    rm[1] = 2 * (b * c + a * d);
    rm[2] = 2 * (b * d - a * c);
    rm[3] = 2 * (b * c - a * d);
    rm[4] = a * a + c * c - b * b - d * d;
    rm[5] = 2 * (c * d + a * b);
    rm[6] = 2 * (b * d + a * c);
    rm[7] = 2 * (c * d - a * b);
    rm[8] = a * a + d * d - b * b - c * c;
}

int generate_rot_matrix(double *rot_mats, double *angles, size_t n_mats, double a0, double a1,
    double a2)
{
    /* check parameters */
    if (!angles || !rot_mats) {ERROR("compute_rot_matrix: one of the arguments is NULL."); return -1;}
    if (!n_mats) {ERROR("compute_rot_matrix: number of matrices must be positive."); return -1;}   

    size_t rm_dims[2] = {n_mats, 9};
    array rm_arr = new_array(2, rm_dims, sizeof(double), rot_mats);
    line rm_ln = init_line(rm_arr, 1);

    double l = sqrt(SQ(a0) + SQ(a1) + SQ(a2));
    double alpha = acos(a2 / l), betta = atan2(a1, a0);
    double bb = -sin(alpha) * cos(betta), cc = -sin(alpha) * sin(betta), dd = -cos(alpha);

    for (int i = 0; i < (int)n_mats; i++)
    {
        UPDATE_LINE(rm_ln, i);

        calc_rotmat(rm_ln->data, 0.5 * angles[i], bb, cc, dd);
    }

    free(rm_ln); free_array(rm_arr);

    return 0;
}