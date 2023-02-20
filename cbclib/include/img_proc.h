#ifndef IMG_PROC_H
#define IMG_PROC_H
#include "include.h"

typedef float (*line_profile)(float err, float wd);

static inline float tophat_profile(float err, float wd)
{
    return fminf(fmaxf(wd - fabsf(err), 0.0f), 1.0f);
}

static inline float linear_profile(float err, float wd)
{
    return fmaxf(1.0f - fabsf(err) / wd, 0.0f);
}

static inline float quad_profile(float err, float wd)
{
    return fmaxf(1.0f - powf(fabsf(err) / wd, 2.0f), 0.0f);
}

#define GS_MIN 0.01831563888873418
#define GS_DIV 1.018657360363774

static inline float gauss_profile(float err, float wd)
{
    return GS_DIV * fmaxf(exp(-SQ(err) / (0.25 * SQ(wd))) - GS_MIN, 0.0f);
}

/*---------------------------------------------------------------------------
                        Drawing lines routines
---------------------------------------------------------------------------*/

int draw_line_int(unsigned *out, const size_t *dims, unsigned max_val, float *lines, const size_t *ldims,
                  float dilation, line_profile profile);
int draw_line_float(float *out, const size_t *dims, float *lines, const size_t *ldims, float dilation,
                    line_profile profile);
int draw_line_index(unsigned **idx, unsigned **x, unsigned **y, float **val, size_t *n_idxs, const size_t *dims,
                    float *lines, const size_t *ldims, float dilation, line_profile profile);

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
int filter_line(float *olines, unsigned char *proc, float *data, const size_t *dims, float *ilines,
                const size_t *ldims, float threshold, float dilation, line_profile profile);

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
int group_line(float *olines, unsigned char *proc, float *data, const size_t *dims, float *ilines,
               const size_t *ldims, float cutoff, float threshold, float dilation, line_profile profile);

int normalise_line(float *out, float *data, const size_t *dims, float *lines, const size_t *ldims,
                   float dilations[3], line_profile profile);

int refine_line(float *out, float *data, const size_t *dims, float *lines, const size_t *ldims,
                float dilation, line_profile profile);

int compute_euler_angles(double *angles, double *rot_mats, size_t n_mats);
int compute_euler_matrix(double *rot_mats, double *angles, size_t n_mats);
int compute_tilt_angles(double *angles, double *rot_mats, size_t n_mats);
int compute_tilt_matrix(double *rot_mats, double *angles, size_t n_mats);
int compute_rotations(double *rot_mats, double *as, double *bs, size_t n_mats);

/*---------------------------------------------------------------------------
                        Model refinement criterion
---------------------------------------------------------------------------*/

double cross_entropy(unsigned *ij, float *p, unsigned *fidxs, size_t *dims, float **lines, const size_t *ldims,
                     size_t lsize, float dilation, float epsilon, line_profile profile, unsigned threads);

#endif