#include <stdio.h>
#include <stdlib.h>
#include "array.h"
#include "median.h"
#include "lsd.h"

static int test_draw_lines();
static int test_lsd();
static int test_median();

int main(int argc, char *argv[])
{
    return test_lsd();
}

static int test_draw_lines()
{
    size_t X = 32;
    size_t Y = 48;
    size_t n_lines = 1;
    double *lines = (double *)calloc(n_lines * 7, sizeof(double));
    unsigned int *out = (unsigned int *)calloc(X * Y, sizeof(unsigned int));

    if (!lines || !out)
    {
        printf("not enough memory\n");
        return EXIT_FAILURE;
    }

    lines[0] = 10.; lines[1] = 10.; lines[2] = 20.; lines[3] = 20.; lines[4] = 3.5;

    draw_lines(out, X, Y, 1, lines, n_lines, 0);

    printf("Result:\n");
    printf("%3d ", 666);
    for (int j = 0; j < (int)X; j++) printf("%3d ", j);
    printf("\n");
    for (int i = 0; i < (int)Y; i++)
    {
        printf("%3d ", i);
        for (int j = 0; j < (int)X; j++) printf("%03d ", out[j + X * i]);
        printf("\n");
    }
    printf("\n");

    free(lines); free(out);

    return EXIT_SUCCESS;
}

static int test_lsd()
{
    double *image;
    double *out;
    int x, y, i, j, n;
    int X = 128;  /* x image size */
    int Y = 128;  /* y image size */

    /* create a simple image: left half black, right half gray */
    image = (double *)malloc(X * Y * sizeof(double));
    if (image == NULL)
    {
        fprintf(stderr,"error: not enough memory\n");
        exit(EXIT_FAILURE);
    }
    for(x = 0; x < X; x++)
    {
        for(y = 0; y < Y; y++)
        {
            image[x + y * X] = x < X / 2 ? 0.0 : 64.; /* image(x,y) */
        }
    }

    /* LSD call */
    lsd(&out, &n, image, X, Y);

    /* print output */
    printf("%d line segments found:\n",n);
    for(i = 0; i < n; i++)
    {
        for(j = 0; j < 7; j++) printf("%f ", out[7 * i + j]);
        printf("\n");
    }

    /* free memory */
    free(image);
    free(out);

    return EXIT_SUCCESS;
}

static int test_median()
{
    size_t X = 30;
    size_t Y = 30;
    size_t dims[2] = {Y, X};
    size_t size = 6; 
    size_t fsize[2] = {size, size};

    double *data = (double *)malloc(X * Y * sizeof(double));
    unsigned char *mask = (unsigned char *)malloc(X * Y * sizeof(unsigned char));
    double *out = (double *)malloc(X * Y * sizeof(double));
    unsigned char *fmask = (unsigned char *)calloc(size * size, sizeof(unsigned char));

    if (!data || !mask || !out || !fmask)
    {
        printf("not enough memory\n");
        return EXIT_FAILURE;
    }

    for (int i = 0; i < (int)Y; i++)
    {
        for (int j = 0; j < (int)X; j++)
        {
            data[j + X * i] = 2 * (pow(i - ((double)Y - 1) / 2, 2.0) + pow(j - ((double)X - 1) / 2, 2.0)) / X / Y;
            mask[j + X * i] = 1;
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
    median_filter(out, data, mask, 2, dims, sizeof(double), fsize,
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

    free(data); free(mask); free(out); free(fmask);

    return EXIT_SUCCESS;
}