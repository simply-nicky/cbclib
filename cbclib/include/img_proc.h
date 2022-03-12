#ifndef IMG_PROC_H
#define IMG_PROC_H
#include "include.h"

// Line draw
int draw_lines(unsigned int *out, size_t Y, size_t X, unsigned int max_val, float *lines,
    size_t n_lines, unsigned int dilation);
int draw_line_indices(unsigned int **out, size_t *n_idxs, size_t Y, size_t X, unsigned int max_val,
    float *lines, size_t n_lines, unsigned int dilation);

// Collapse adjacent lines into one and filter out the bad ones
int filter_lines(float *olines, float *data, size_t Y, size_t X, float *ilines, size_t n_lines,
    float x_c, float y_c, float* radii, float thr);

int compute_euler_angles(double *eulers, double *rot_mats, size_t n_mats);
int compute_rot_matrix(double *rot_mats, double *eulers, size_t n_mats);
int generate_rot_matrix(double *rot_mats, double *angles, size_t n_mats, double a0, double a1,
    double a2);

#endif