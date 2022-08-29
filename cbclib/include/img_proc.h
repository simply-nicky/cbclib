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

/*---------------------------------------------------------------------------
                        Drawing lines routines
---------------------------------------------------------------------------*/

int draw_lines(unsigned int *out, size_t Y, size_t X, unsigned int max_val, float *lines,
    size_t *ldims, float dilation, line_profile profile);
int draw_line_indices(unsigned int **out, size_t *n_idxs, size_t Y, size_t X, unsigned int max_val,
    float *lines, size_t *ldims, float dilation, line_profile profile);

/*-------------------------------------------------------------------------------*/
/** Filter lines with an image moment MO less then a threshold.

    @param olines       Output buffer of lines of shape ldims.
    @param proc         Buffer of flags. If proc[i] is 0, than the given line was
                        discarded.
    @param data         Input image of shape (Y, X).
    @param Y            Image size along the vertical axis.
    @param X            Image size along the horizontal axis.
    @param ilines       Input buffer of lines. If olines is uninitialized, the data
                        from ilines is copied to olines.
    @param ldims        Shape of a ilines buffer.
    @param threshold    Filtering threshold. The line is discarded if 0-th moment
                        M0 < threshold.
    @param dilation     Line mask dilation in pixels.

    @return             Returns 0 if it finished normally, 1 otherwise.
 */
int filter_lines(float *olines, unsigned char *proc, float *data, size_t Y, size_t X, float *ilines,
    size_t *ldims, float threshold, float dilation);

/*-------------------------------------------------------------------------------*/
/** Group a pair of lines into one if the correlation if above the treshold.

    @param olines       Output buffer of lines of shape ldims.
    @param proc         Buffer of flags. If proc[i] is 0, than the given line was
                        discarded.
    @param data         Input image of shape (Y, X).
    @param Y            Image size along the vertical axis.
    @param X            Image size along the horizontal axis.
    @param ilines       Input buffer of lines. If olines is uninitialized, the data
                        from ilines is copied to olines.
    @param ldims        Shape of a ilines buffer.
    @param threshold    Grouping threshold. A pair of lines is merged if the 
                        correlation[i, j] > threshold.
    @param dilation     Line mask dilation in pixels.

    @return             Returns 0 if it finished normally, 1 otherwise.
 */
int group_lines(float *olines, unsigned char *proc, float *data, size_t Y, size_t X, float *ilines,
    size_t *ldims, float cutoff, float threshold, float dilation);

int compute_euler_angles(double *eulers, double *rot_mats, size_t n_mats);
int compute_euler_matrix(double *rot_mats, double *eulers, size_t n_mats);
int compute_tilt_matrix(double *rot_mats, double *angles, size_t n_mats, double a0, double a1, double a2);

#endif