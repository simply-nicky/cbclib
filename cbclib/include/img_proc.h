#ifndef IMG_PROC_H
#define IMG_PROC_H
#include "include.h"

typedef int (*line_profile)(int max_val, float err, float wd);

static inline int tophat_profile(int max_val, float err, float wd)
{
    return max_val * (1.0f - fmaxf((fabsf(err) - wd + 1.0f), 0.0f));
}

static inline int linear_profile(int max_val, float err, float wd)
{
    return max_val * (1.0f - fminf(fabsf(err) / wd, 1.0f));
}

static inline int quad_profile(int max_val, float err, float wd)
{
    return max_val * (1.0f - powf(fminf(fabsf(err) / wd, 1.0f), 2.0));
}

// Line draw
int draw_lines(unsigned int *out, size_t Y, size_t X, unsigned int max_val, float *lines,
    size_t *ldims, float dilation, line_profile profile);
int draw_line_indices(unsigned int **out, size_t *n_idxs, size_t Y, size_t X, unsigned int max_val,
    float *lines, size_t *ldims, float dilation, line_profile profile);

// Collapse adjacent lines into one and filter out the bad ones
int filter_lines(float *olines, unsigned char *proc, float *data, size_t Y, size_t X, float *ilines,
    size_t *ldims, float threshold, float dilation);
int group_lines(float *olines, unsigned char *proc, float *data, size_t Y, size_t X, float *ilines,
    size_t *ldims, float cutoff, float threshold, float dilation);

int compute_euler_angles(double *eulers, double *rot_mats, size_t n_mats);
int compute_euler_matrix(double *rot_mats, double *eulers, size_t n_mats);
int compute_tilt_matrix(double *rot_mats, double *angles, size_t n_mats, double a0, double a1, double a2);

#endif