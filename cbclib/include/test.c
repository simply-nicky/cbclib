#include <stdio.h>
#include <stdlib.h>
#include "median.h"

int main(int argc, char *argv[])
{
    int X = 16;
    int Y = 16;
    double *data = (double *)malloc(X * Y * sizeof(double));
    double *out = (double *)malloc(X * Y * sizeof(double));

    if (!data || !out)
    {
        printf("not enough memory\n");
        return EXIT_FAILURE;
    }
    for (int i = 0; i < X * Y; i++) data[i] = (double)i;

    size_t dims[2] = {X, Y};
    size_t fsize[2] = {3, 3};
    double cval = 0.0;

    median_filter(out, data, 2, dims, sizeof(double), fsize,
        EXTEND_MIRROR, &cval, compare_double, 1);

    printf("Result:\n");
    for (int i = 0; i < X; i++)
    {
        for (int j = 0; j < Y; j++) printf("%.0f ", out[i]);
        printf("\n");
    }
    printf("\n");

    return EXIT_SUCCESS;
}