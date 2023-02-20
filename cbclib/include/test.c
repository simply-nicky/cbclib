#include <stdio.h>
#include <stdlib.h>
#include "array.h"
#include "median.h"
#include "lsd.h"
#include "img_proc.h"
#include "sgn_proc.h"
#include "kd_tree.h"

static int test_draw_line();
static int test_draw_line_index();
static int test_lsd();
static int test_median();
static int test_pairs();
static int test_kerreg();
static int test_kd_tree();

int main(int argc, char *argv[])
{
    return test_kd_tree();
}

static int test_draw_line()
{
    size_t dims[2] = {48, 32};
    size_t n_lines = 1;
    size_t ldims[2] = {n_lines, 7};

    float *lines = MALLOC(float, n_lines * 7);
    unsigned int *out = (unsigned int *)calloc(dims[0] * dims[1], sizeof(unsigned int));

    if (!lines || !out)
    {
        printf("not enough memory\n");
        return EXIT_FAILURE;
    }

    lines[0] = 10.; lines[1] = 10.; lines[2] = 20.; lines[3] = 20.; lines[4] = 3.5;

    draw_line(out, dims, 1, lines, ldims, 0.0, linear_profile);

    printf("Result:\n");
    printf("%3d ", 666);
    for (int j = 0; j < (int)dims[1]; j++) printf("%3d ", j);
    printf("\n");
    for (int i = 0; i < (int)dims[0]; i++)
    {
        printf("%3d ", i);
        for (int j = 0; j < (int)dims[1]; j++) printf("%03d ", out[j + dims[1] * i]);
        printf("\n");
    }
    printf("\n");

    DEALLOC(lines); DEALLOC(out);

    return EXIT_SUCCESS;
}

static int test_draw_line_index()
{
    size_t dims = {48, 32};
    size_t n_lines = 2;
    size_t ldims[2] = {n_lines, 7};

    float *lines = (float *)calloc(n_lines * 7, sizeof(float));
    unsigned int *out;
    size_t n_idxs;

    if (!lines)
    {
        printf("not enough memory\n");
        return EXIT_FAILURE;
    }

    lines[0] = 10.; lines[1] = 10.; lines[2] = 20.; lines[3] = 20.; lines[4] = 3.5;
    lines[7] = 30.; lines[8] = 15.; lines[9] = 5.; lines[10] = 10.; lines[11] = 3.5;

    draw_line_index(&out, &n_idxs, dims, 255, lines, ldims, 0.0, linear_profile);

    printf("Result:\n");
    printf("idx  x   y   I \n");
    for (int i = 0; i < (int)n_idxs; i++)
    {
        for (int j = 0; j < 4; j++) printf("%03d ", out[j + 4 * i]);
        printf("\n");
    }
    printf("\n");

    DEALLOC(lines); DEALLOC(out);

    return EXIT_SUCCESS;
}

static int test_lsd()
{
    float *image;
    float *out;
    int x, y, i, j, n;
    size_t dims[2] = {128, 128};  /* Y, X */

    /* create a simple image: left half black, right half gray */
    image = MALLOC(float, dims[0] * dims[1]);
    if (image == NULL)
    {
        fprintf(stderr, "error: not enough memory\n");
        exit(EXIT_FAILURE);
    }
    for(x = 0; x < dims[1]; x++)
    {
        for(y = 0; y < dims[0]; y++)
        {
            image[x + y * dims[1]] = x < dims[1] / 2 ? 0.0 : 64.; /* image(x, y) */
        }
    }

    /* LSD call */
    lsd(&out, &n, image, dims);

    /* print output */
    printf("%d line segments found:\n",n);
    for(i = 0; i < n; i++)
    {
        for(j = 0; j < 7; j++) printf("%f ", out[7 * i + j]);
        printf("\n");
    }

    /* free memory */
    DEALLOC(image);
    DEALLOC(out);

    return EXIT_SUCCESS;
}

static int test_median()
{
    size_t X = 30;
    size_t Y = 30;
    size_t dims[2] = {Y, X};
    size_t size = 6; 
    size_t fsize[2] = {size, size};

    double *data = MALLOC(double, X * Y);
    unsigned char *mask = MALLOC(unsigned char, X * Y);
    unsigned char *imask = MALLOC(unsigned char, X * Y);
    double *out = MALLOC(double, X * Y);
    unsigned char *fmask = MALLOC(unsigned char, size * size);

    if (!data || !mask || !out || !fmask)
    {
        printf("not enough memory\n");
        return EXIT_FAILURE;
    }

    for (int i = 0; i < (int)Y; i++)
    {
        for (int j = 0; j < (int)X; j++)
        {
            data[j + X * i] = 2 * (SQ(i - ((double)Y - 1) / 2) + SQ(j - ((double)X - 1) / 2)) / X / Y;
            mask[j + X * i] = 1; imask[j + X * i] = 1;
        }
    }

    for (int i = 2; i < 4; i++)
    {
        for (int j = 0; j < (int)size; j++) fmask[j + size * i] = 1;
    }

    printf("Input:\n");
    printf("%4d ", 666);
    for (int j = 0; j < (int)X; j++) printf("%4d ", j);
    printf("\n");
    for (int i = 0; i < (int)Y; i++)
    {
        printf("%4d ", i);
        for (int j = 0; j < (int)X; j++) printf("%0.2f ", data[j + X * i]);
        printf("\n");
    }
    printf("\n");


    double cval = 0.0;
    median_filter(out, data, mask, imask, 2, dims, sizeof(double), fsize,
                  fmask, EXTEND_REFLECT, &cval, compare_uint, 1);

    printf("Result:\n");
    printf("%4d ", 666);
    for (int j = 0; j < (int)X; j++) printf("%4d ", j);
    printf("\n");
    for (int i = 0; i < (int)Y; i++)
    {
        printf("%4d ", i);
        for (int j = 0; j < (int)X; j++) printf("%0.2f ", out[j + X * i]);
        printf("\n");
    }
    printf("\n");

    DEALLOC(data); DEALLOC(mask); DEALLOC(imask); DEALLOC(out); DEALLOC(fmask);

    return EXIT_SUCCESS;
}

static const float LINES[14] = 
   {1.0, 2.0, 1.0, 2.0, 0.0, 0.0, 0.0,
    1.0, 2.0, 2.0, 3.0, 1.0, 0.0, 0.0,};

static int test_pairs()
{
    size_t dims[2] = {10, 10};
    int n_lines = 2;
    size_t ldims[2] = {n_lines, 7};

    float *data = MALLOC(float, dims[0] * dims[1]);
    float *ilines = (float *)LINES;
    unsigned char *proc = MALLOC(unsigned char, n_lines);
    for (int i = 0; i < n_lines; i++) proc[i] = 1;
    float *olines = MALLOC(float, n_lines * 7);
    for (int i = 0; i < dims[0] * dims[1]; i++) data[i] = 1.0;

    filter_line(olines, proc, data, dims, ilines, ldims, 1.0, 0.0, gauss_profile);

    int out_lines = 0;
    for (int i = 0; i < n_lines; i++) out_lines += proc[i];
    printf("%3d lines remained.\n", out_lines);

    DEALLOC(data); DEALLOC(proc); DEALLOC(olines);

    return 0;
}

static int test_kerreg()
{
    size_t npts = 1, ndim = 3, nhat = 100;

    double * x = MALLOC(double, npts * ndim);
    double * w = MALLOC(double, npts);
    double * y = MALLOC(double, npts);
    double * xhat = MALLOC(double, nhat * ndim);
    double * yhat = MALLOC(double, nhat);

    srand(time(NULL));
    for (int i = 0; i < (int)npts; i++) {y[i] = rand() / (double)RAND_MAX; w[i] = 1.0;}
    for (int i = 0; i < (int)npts * ndim; i++) x[i] = rand() / (double)RAND_MAX;
    for (int i = 0; i < (int)nhat * ndim; i++) xhat[i] = 2.0 * (rand() / (double)RAND_MAX - 0.5);

    predict_kerreg(y, w, x, npts, ndim, yhat, xhat, nhat, rbf, 1e-2, 1e-1, 1);
    predict_kerreg(y, w, x, npts, ndim, yhat, xhat, nhat, rbf, 1e-2, 1e-1, 1);

    printf("Success.\n");

    DEALLOC(x); DEALLOC(y); DEALLOC(xhat); DEALLOC(yhat);

    return 0;
}

static int test_kd_tree()
{
    float pos[20] = {35.0, 60.0, 5.0, 45.0, 65.0, 80.0, 0.0, 55.0, 85.0, 40.0,
                     25.0, 20.0, 50.0, 30.0, 90.0, 75.0, 70.0, 15.0, 95.0, 10.0};
    size_t idxs[20] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
    fprintf(stdout, "Building the tree\n");
    kd_tree tree = kd_build(pos, 10, 2, idxs, sizeof(size_t));
    fprintf(stdout, "\nPrinting the tree\n\n");
    kd_print(stdout, tree);

    fprintf(stdout, "\nPrinting the cell: (%.2f %.2f %.2f %.2f)\n",
            tree->rect->min[0], tree->rect->min[1], tree->rect->max[0], tree->rect->max[1]);

    int axis = 0;
    kd_node node = kd_find_min(tree, axis);
    fprintf(stdout, "\nMinimum along axis %d: (%.2f, %.2f)\n", axis, node->pos[0], node->pos[1]);

    fprintf(stdout, "\nDeleting a point (%.2f %.2f)\n", node->pos[0], node->pos[1]);
    kd_delete(tree, node->pos);

    fprintf(stdout, "\nPrinting the tree\n\n");
    kd_print(stdout, tree);

    fprintf(stdout, "\nPrinting the cell: (%.2f %.2f %.2f %.2f)\n",
            tree->rect->min[0], tree->rect->min[1], tree->rect->max[0], tree->rect->max[1]);

    kd_query query = kd_find_nearest(tree, node->pos);
    fprintf(stdout, "\nThe nearest neighbour at distance %.2f: (%.2f %.2f)\n",
            sqrtf(query->dist), query->node->pos[0], query->node->pos[1]);

    float range = 40.0f;
    query_stack stack = kd_find_range(tree, node->pos, range);
    fprintf(stdout, "\nPoints in a %.2f range:\n", stack, range);
    for (query_stack node = stack; node; node = node->next)
    {
        fprintf(stdout, "(%.2f %.2f) at distance %.2f\n",
                node->query->node->pos[0], node->query->node->pos[1], sqrtf(node->query->dist));
    }

    kd_free(tree);
    return 0;
}