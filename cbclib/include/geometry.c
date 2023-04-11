#include "geometry.h"
#include "array.h"

#define TOL 3.1425926535897937e-05

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

    if (n_mats == 0) return 0;

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

    if (n_mats == 0) return 0;

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

    if (n_mats == 0) return 0;

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

    if (n_mats == 0) return 0;

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
    
    if (n_mats == 0) return 0;

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

int det2k(double *karr, double *x, double *y, unsigned *idxs, size_t ksize, double *src, unsigned threads)
{
    if (!karr || !x || !y || !idxs || !src) {ERROR("det2k: one of the arguments is NULL."); return -1;}

    if (ksize == 0) return 0;

    #pragma omp parallel num_threads(threads)
    {
        int i;
        double dx, dy, phi, theta;

        #pragma omp for
        for (i = 0; i < (int)ksize; i++)
        {
            dx = x[i] - src[3 * idxs[i]];
            dy = y[i] - src[3 * idxs[i] + 1];
            phi = atan2(dy, dx);
            theta = acos(-src[3 * idxs[i] + 2] / sqrt(SQ(dx) + SQ(dy) + SQ(src[3 * idxs[i] + 2])));
            karr[3 * i] = sin(theta) * cos(phi);
            karr[3 * i + 1] = sin(theta) * sin(phi);
            karr[3 * i + 2] = cos(theta);
        }
    }

    return 0;
}

int k2det(double *x, double *y, double *karr, unsigned *idxs, size_t ksize, double *src, unsigned threads)
{
    if (!karr || !x || !y || !idxs || !src) {ERROR("k2det: one of the arguments is NULL."); return -1;}

    if (ksize == 0) return 0;

    #pragma omp parallel num_threads(threads)
    {
        int i;
        double dz, phi, theta;

        #pragma omp for
        for (i = 0; i < (int)ksize; i++)
        {
            phi = atan2(karr[3 * i + 1], karr[3 * i]);
            theta = acos(karr[3 * i + 2] / sqrt(SQ(karr[3 * i]) + SQ(karr[3 * i + 1]) + SQ(karr[3 * i + 2])));
            dz = src[3 * idxs[i] + 2] * tan(theta);
            x[i] = src[3 * idxs[i]] - dz * cos(phi);
            y[i] = src[3 * idxs[i] + 1] - dz * sin(phi);
        }
    }

    return 0;
}

int k2smp(double *pts, double *karr, unsigned *idxs, size_t ksize, double *z, double *src, unsigned threads)
{
    if (!pts || !karr || !idxs || !z || !src) {ERROR("k2det: one of the arguments is NULL."); return -1;}

    if (ksize == 0) return 0;

    #pragma omp parallel num_threads(threads)
    {
        int i;
        double dz, phi, theta;

        #pragma omp for
        for (i = 0; i < (int)ksize; i++)
        {
            phi = atan2(karr[3 * i + 1], karr[3 * i]);
            theta = acos(karr[3 * i + 2] / sqrt(SQ(karr[3 * i]) + SQ(karr[3 * i + 1]) + SQ(karr[3 * i + 2])));
            dz = (z[idxs[i]] - src[2]) * tan(theta);
            pts[3 * i    ] = src[0] + dz * cos(phi);
            pts[3 * i + 1] = src[1] + dz * sin(phi);
            pts[3 * i + 2] = z[idxs[i]];
        }
    }

    return 0;
}

int rotate_vec(double *out, double *vecs, unsigned *idxs, size_t vsize, double *rmats, unsigned threads)
{
    if (!out || !vecs || !idxs || !rmats) {ERROR("rotate_vec: one of the arguments is NULL."); return -1;}

    if (vsize == 0) return 0;

    #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < (int)vsize; i++)
    {
        out[3 * i    ] = rmats[9 * idxs[i]    ] * vecs[3 * i] + rmats[9 * idxs[i] + 1] * vecs[3 * i + 1] + rmats[9 * idxs[i] + 2] * vecs[3 * i + 2];
        out[3 * i + 1] = rmats[9 * idxs[i] + 3] * vecs[3 * i] + rmats[9 * idxs[i] + 4] * vecs[3 * i + 1] + rmats[9 * idxs[i] + 5] * vecs[3 * i + 2];
        out[3 * i + 2] = rmats[9 * idxs[i] + 6] * vecs[3 * i] + rmats[9 * idxs[i] + 7] * vecs[3 * i + 1] + rmats[9 * idxs[i] + 8] * vecs[3 * i + 2];
    }

    return 0;
}