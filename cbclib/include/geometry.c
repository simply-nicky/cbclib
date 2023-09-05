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

static void rotmat_to_euler(float *ang, float *rm)
{
    ang[1] = acosf(rm[8]);
    if (ang[1] < 1e-8) {ang[0] = atan2f(-rm[3], rm[0]); ang[2] = 0.0f; }
    else if (M_PI - ang[1] < TOL)
    {ang[0] = atan2f(rm[3], rm[0]); ang[2] = 0.0f; }
    else {ang[0] = atan2f(rm[6], -rm[7]); ang[2] = atan2f(rm[2], rm[5]); }
    if (ang[0] < 0.0f) ang[0] += M_2__PI;
    if (ang[2] < 0.0f) ang[2] += M_2__PI;
}

int compute_euler_angles(float *angles, float *rot_mats, size_t n_mats)
{
    /* check parameters */
    if (!angles || !rot_mats) {ERROR("compute_euler_angles: one of the arguments is NULL."); return -1;}

    if (n_mats == 0) return 0;

    size_t rm_dims[2] = {n_mats, 9};
    array rm_arr = new_array(2, rm_dims, sizeof(float), rot_mats);
    line rm_ln = init_line(rm_arr, 1);

    size_t a_dims[2] = {n_mats, 3};
    array ang_arr = new_array(2, a_dims, sizeof(float), angles);
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

static void euler_to_rotmat(float *rm, float *ang)
{
    float c0 = cosf(ang[0]), c1 = cosf(ang[1]), c2 = cosf(ang[2]);
    float s0 = sinf(ang[0]), s1 = sinf(ang[1]), s2 = sinf(ang[2]);

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

int compute_euler_matrix(float *rot_mats, float *angles, size_t n_mats)
{
    /* check parameters */
    if (!angles || !rot_mats) {ERROR("compute_euler_matrix: one of the arguments is NULL."); return -1;}

    if (n_mats == 0) return 0;

    size_t rm_dims[2] = {n_mats, 9};
    array rm_arr = new_array(2, rm_dims, sizeof(float), rot_mats);
    line rm_ln = init_line(rm_arr, 1);

    size_t a_dims[2] = {n_mats, 3};
    array ang_arr = new_array(2, a_dims, sizeof(float), angles);
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

static void rotmat_to_tilt(float *ang, float *rm)
{
    float a0 = rm[7] - rm[5];
    float a1 = rm[2] - rm[6];
    float a2 = rm[3] - rm[1];
    float l = sqrtf(SQ(a0) + SQ(a1) + SQ(a2));
    ang[0] = acosf(0.5f * (rm[0] + rm[4] + rm[8] - 1.0f));
    ang[1] = acosf(a2 / l);
    ang[2] = atan2f(a1, a0);
}

int compute_tilt_angles(float *angles, float *rot_mats, size_t n_mats)
{
    /* check parameters */
    if (!angles || !rot_mats) {ERROR("compute_tilt_angles: one of the arguments is NULL."); return -1;}

    if (n_mats == 0) return 0;

    size_t rm_dims[2] = {n_mats, 9};
    array rm_arr = new_array(2, rm_dims, sizeof(float), rot_mats);
    line rm_ln = init_line(rm_arr, 1);

    size_t a_dims[2] = {n_mats, 3};
    array ang_arr = new_array(2, a_dims, sizeof(float), angles);
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

static void tilt_to_rotmat(float *rm, float *ang)
{
    float a = cosf(0.5f * ang[0]), b = -sin(ang[1]) * cosf(ang[2]) * sinf(0.5f * ang[0]);
    float c = -sin(ang[1]) * sinf(ang[2]) * sinf(0.5f * ang[0]), d = -cos(ang[1]) * sinf(0.5f * ang[0]);

    rm[0] = a * a + b * b - c * c - d * d;
    rm[1] = 2.0f * (b * c + a * d);
    rm[2] = 2.0f * (b * d - a * c);
    rm[3] = 2.0f * (b * c - a * d);
    rm[4] = a * a + c * c - b * b - d * d;
    rm[5] = 2.0f * (c * d + a * b);
    rm[6] = 2.0f * (b * d + a * c);
    rm[7] = 2.0f * (c * d - a * b);
    rm[8] = a * a + d * d - b * b - c * c;
}

int compute_tilt_matrix(float *rot_mats, float *angles, size_t n_mats)
{
    /* check parameters */
    if (!angles || !rot_mats) {ERROR("compute_tilt_matrix: one of the arguments is NULL."); return -1;}

    if (n_mats == 0) return 0;

    size_t rm_dims[2] = {n_mats, 9};
    array rm_arr = new_array(2, rm_dims, sizeof(float), rot_mats);
    line rm_ln = init_line(rm_arr, 1);

    size_t a_dims[2] = {n_mats, 3};
    array ang_arr = new_array(2, a_dims, sizeof(float), angles);
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

    @note           Yields nan, if cosf(a, b) = -1.0f.
 */
static void rotation_of_a_to_b(float *rm, float *a, float *b)
{
    float vx = a[1] * b[2] - a[2] * b[1];
    float vy = a[2] * b[0] - a[0] * b[2];
    float vz = a[0] * b[1] - a[1] * b[0];
    float rat = 1.0f / (1.0f + a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);
    rm[0] = 1.0f - rat * (SQ(vy) + SQ(vz));
    rm[1] = -vz + rat * vx * vy;
    rm[2] =  vy + rat * vx * vz;
    rm[3] =  vz + rat * vx * vy;
    rm[4] = 1.0f - rat * (SQ(vx) + SQ(vz));
    rm[5] = -vx + rat * vy * vz;
    rm[6] = -vy + rat * vx * vz;
    rm[7] =  vx + rat * vy * vz;
    rm[8] = 1.0f - rat * (SQ(vx) + SQ(vy));
}

int compute_rotations(float *rot_mats, float *as, float *bs, size_t n_mats)
{
    /* check parameters */
    if (!as || !bs || !rot_mats) {ERROR("compute_rotations: one of the arguments is NULL."); return -1;}
    
    if (n_mats == 0) return 0;

    size_t rm_dims[2] = {n_mats, 9};
    array rm_arr = new_array(2, rm_dims, sizeof(float), rot_mats);
    line rm_ln = init_line(rm_arr, 1);
    float a[3], b[3];
    float a_abs, b_abs;

    for (int i = 0; i < (int)n_mats; i++)
    {
        UPDATE_LINE(rm_ln, i);

        a_abs = sqrtf(SQ(as[3 * i]) + SQ(as[3 * i + 1]) + SQ(as[3 * i + 2]));
        a[0] = as[3 * i] / a_abs; a[1] = as[3 * i + 1] / a_abs; a[2] = as[3 * i + 2] / a_abs;
        b_abs = sqrtf(SQ(bs[3 * i]) + SQ(bs[3 * i + 1]) + SQ(bs[3 * i + 2]));
        b[0] = bs[3 * i] / b_abs; b[1] = bs[3 * i + 1] / b_abs; b[2] = bs[3 * i + 2] / b_abs;
        rotation_of_a_to_b(rm_ln->data, a, b);
    }

    DEALLOC(rm_ln); free_array(rm_arr);

    return 0;
}

int rotate_vec(float *out, float *vecs, unsigned *idxs, size_t vsize, float *rmats, unsigned threads)
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

/*----------------------------------------------------------------------------*/
/*---------- Conversions from the real space to the reciprocal ---------------*/
/*----------------------------------------------------------------------------*/
int det2k(float *karr, float *x, float *y, unsigned *idxs, size_t ksize, float *src, unsigned threads)
{
    if (!karr || !x || !y || !idxs || !src) {ERROR("det2k: one of the arguments is NULL."); return -1;}

    if (ksize == 0) return 0;

    float dist;
    #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < (int)ksize; i++)
    {
        dist = sqrtf(SQ(x[i] - src[3 * idxs[i]]) + SQ(y[i] - src[3 * idxs[i] + 1]) + SQ(src[3 * idxs[i] + 2]));
        karr[3 * i + 0] = (x[i] - src[3 * idxs[i]]) / dist;
        karr[3 * i + 1] = (y[i] - src[3 * idxs[i] + 1]) / dist;
        karr[3 * i + 2] = -src[3 * idxs[i] + 2] / dist;
    }

    return 0;
}

int det2k_vjp(float *xout, float *yout, float *sout, float *vec, float *x, float *y, unsigned *idxs,
              size_t ksize, float *src, size_t ssize, unsigned threads)
{
    if (!xout || !sout || !vec || !x || !y || !idxs || !src)
    {ERROR("det2k_vjp: one of the arguments is NULL."); return -1;}

    if (ksize == 0) return 0;

    #pragma omp parallel num_threads(threads)
    {
        int i;
        float *sbuf = (float *)calloc(3 * ssize, sizeof(float));
        float dist, dsq, prd;

        #pragma omp for
        for (i = 0; i < (int)ksize; i++)
        {
            dsq = SQ(x[i] - src[3 * idxs[i]]) + SQ(y[i] - src[3 * idxs[i] + 1]) + SQ(src[3 * idxs[i] + 2]);
            dist = sqrtf(dsq);
            prd = vec[3 * i] * (x[i] - src[3 * idxs[i]]) + vec[3 * i + 1] * (y[i] - src[3 * idxs[i] + 1])
                - vec[3 * i + 2] * src[3 * idxs[i] + 2];
            xout[i] = vec[3 * i] / dist - prd * (x[i] - src[3 * idxs[i]]) / (dist * dsq);
            yout[i] = vec[3 * i + 1] / dist - prd * (y[i] - src[3 * idxs[i] + 1]) / (dist * dsq);

            sbuf[3 * idxs[i]] += -vec[3 * i] / dist + prd * (x[i] - src[3 * idxs[i]]) / (dist * dsq);
            sbuf[3 * idxs[i] + 1] += -vec[3 * i + 1] / dist + prd * (y[i] - src[3 * idxs[i] + 1]) / (dist * dsq);
            sbuf[3 * idxs[i] + 2] += -1.0f / dist + prd * src[3 * idxs[i] + 2] / (dist * dsq);
        }

        for (i = 0; i < (int)(3 * ssize); i++)
        {
            #pragma omp atomic
            sout[i] += sbuf[i];
        }

        DEALLOC(sbuf);
    }

    return 0;
}

int k2det(float *x, float *y, float *karr, unsigned *idxs, size_t ksize, float *src, unsigned threads)
{
    if (!karr || !x || !y || !idxs || !src) {ERROR("k2det: one of the arguments is NULL."); return -1;}

    if (ksize == 0) return 0;

    #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < (int)ksize; i++)
    {
        x[i] = src[3 * idxs[i] + 0] - karr[3 * i + 0] / karr[3 * i + 2] * src[3 * idxs[i] + 2];
        y[i] = src[3 * idxs[i] + 1] - karr[3 * i + 1] / karr[3 * i + 2] * src[3 * idxs[i] + 2];
    }

    return 0;
}

int k2det_vjp(float *kout, float *sout, float *xvec, float *yvec, float *karr, unsigned *idxs, size_t ksize,
              float *src, size_t ssize, unsigned threads)
{
    if (!kout || !sout || !xvec || !yvec || !karr || !idxs || !src)
    {ERROR("k2det_vjp: one of the arguments is NULL."); return -1;}

    if (ksize == 0) return 0;

    #pragma omp parallel num_threads(threads)
    {
        int i;
        float *sbuf = (float *)calloc(3 * ssize, sizeof(float));

        #pragma omp for
        for (i = 0; i < (int)ksize; i++)
        {
            kout[3 * i + 0] = -xvec[i] / karr[3 * i + 2] * src[3 * idxs[i] + 2];
            kout[3 * i + 1] = -yvec[i] / karr[3 * i + 2] * src[3 * idxs[i] + 2];
            kout[3 * i + 2] = (xvec[i] * karr[3 * i]+ yvec[i] * karr[3 * i + 1])
                            * src[3 * idxs[i] + 2] / SQ(karr[3 * i + 2]);

            sbuf[3 * idxs[i]] += xvec[i];
            sbuf[3 * idxs[i] + 1] += yvec[i];
            sbuf[3 * idxs[i] + 2] += -(xvec[i] * karr[3 * i] + yvec[i] * karr[3 * i + 1]) / karr[3 * i + 2];
        }

        for (i = 0; i < (int)(3 * ssize); i++)
        {
            #pragma omp atomic
            sout[i] += sbuf[i];
        }

        DEALLOC(sbuf);
    }

    return 0;
}

int k2smp(float *pts, float *karr, unsigned *idxs, size_t ksize, float *z, float *src, unsigned threads)
{
    if (!pts || !karr || !idxs || !z || !src) {ERROR("k2smp: one of the arguments is NULL."); return -1;}

    if (ksize == 0) return 0;

    #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < (int)ksize; i++)
    {
        pts[3 * i + 0] = src[0] + karr[3 * i + 0] / karr[3 * i + 2] * (z[idxs[i]] - src[2]);
        pts[3 * i + 1] = src[1] + karr[3 * i + 1] / karr[3 * i + 2] * (z[idxs[i]] - src[2]);
        pts[3 * i + 2] = z[idxs[i]];
    }

    return 0;
}

int k2smp_vjp(float *kout, float *zout, float *sout, float *vec, float *karr, unsigned *idxs, size_t ksize,
              float *z, size_t zsize, float *src, unsigned threads)
{
    if (!kout || !zout || !sout || !vec || !karr || !idxs || !z || !src)
    {ERROR("k2smp_vjp: one of the arguments is NULL."); return -1;}

    if (ksize == 0) return 0;

    #pragma omp parallel num_threads(threads)
    {
        int i;
        float *zbuf = (float *)calloc(zsize, sizeof(float));
        float sbuf[3] = {0.0f, 0.0f, 0.0f};

        #pragma omp for
        for (i = 0; i < (int)ksize; i++)
        {
            kout[3 * i + 0] = vec[3 * i + 0] / karr[3 * i + 2] * (z[idxs[i]] - src[2]);
            kout[3 * i + 1] = vec[3 * i + 1] / karr[3 * i + 2] * (z[idxs[i]] - src[2]);
            kout[3 * i + 2] = (vec[3 * i] * karr[3 * i] + vec[3 * i + 1] * karr[3 * i + 1])
                            * (src[2] - z[idxs[i]]) / SQ(karr[3 * i + 2]);

            zbuf[idxs[i]] += (vec[3 * i] * karr[3 * i] + vec[3 * i + 1] * karr[3 * i + 1])
                           / karr[3 * i + 2] + vec[3 * i + 2];

            sbuf[0] += vec[3 * i]; sbuf[1] += vec[3 * i + 1];
            sbuf[2] += -(vec[3 * i] * karr[3 * i] + vec[3 * i + 1] * karr[3 * i + 1]) / karr[3 * i + 2];
        }

        for (i = 0; i < (int)zsize; i++)
        {
            #pragma omp atomic
            zout[i] += zbuf[i];
        }
        for (i = 0; i < 3; i++)
        {
            #pragma omp atomic
            sout[i] += sbuf[i];
        }

        DEALLOC(zbuf);
    }

    return 0;
}

/*----------------------------------------------------------------------------*/
/*------------------ Finding source lines for CBD model ----------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------
    Solving a quadratic equation:

    f1 = o . q - s . q          f2 = q . e

    a * t^2 - 2 b * t + c = 0
    a = f2^2 + q_z^2     b = f1 * f2
    c = f1^2 - (1 - s^2) * q_z^2
-----------------------------------------------------------------*/
static int find_intersection(float *t, float *q, float *e, float *s, float *lim)
{
    /* oq is given by: oq = o . q = - q^2 / 2 */
    float oq = -0.5f * (SQ(q[0]) + SQ(q[1]) + SQ(q[2]));
    float f1 = oq - s[0] * q[0] - s[1] * q[1];
    float f2 = e[0] * q[0] + e[1] * q[1];

    float a = SQ(f2) + SQ(q[2]);
    float b = f1 * f2;
    float c = SQ(f1) - (1.0f - SQ(s[0]) - SQ(s[1])) * SQ(q[2]);

    int sign = 0;
    float res, x, y, prd;
    if (SQ(b) > a * c)
    {
        float delta = sqrtf(SQ(b) - a * c);
        res = (b - delta) / a;
        x = s[0] + res * e[0];
        y = s[1] + res * e[1];
        prd = x * q[0] + y * q[1] + sqrtf(1.0f - SQ(x) - SQ(y)) * q[2] - oq;

        if ((res >= lim[0]) && (res <= lim[1]) && (fabsf(prd) < FLT_EPSILON))
        {
            t[0] = res; sign = -1;
        }

        res = (b + delta) / a;
        x = s[0] + res * e[0];
        y = s[1] + res * e[1];
        prd = x * q[0] + y * q[1] + sqrtf(1.0f - SQ(x) - SQ(y)) * q[2] - oq;

        if ((res >= lim[0]) && (res <= lim[1]) && (fabsf(prd) < FLT_EPSILON))
        {
            t[0] = res; sign = 1;
        }

        return sign;
    }

    return sign;
}

/*----------------------------------------------------------------
    Gradient array is given by:
    [dt / dq[0], dt / dq[1], dt / dq[2], dt / ds[0], dt / ds[1]]
-----------------------------------------------------------------*/
static void find_intersection_vjp(float *qs_ctg, int sign, float t_ctg, float *q, float *e,
                                  float *s, float *lim)
{
    /* oq is given by: oq = o . q = - q^2 / 2 */
    float oq = -0.5f * (SQ(q[0]) + SQ(q[1]) + SQ(q[2]));
    float f1 = oq - s[0] * q[0] - s[1] * q[1];
    float f2 = e[0] * q[0] + e[1] * q[1];

    float a = SQ(f2) + SQ(q[2]);
    float da[5] = {2.0f * f2 * e[0], 2.0f * f2 * e[1], 2.0f * q[2], 0.0f, 0.0f};

    float b = f1 * f2;
    float db[5] = {e[0] * f1 - f2 * (q[0] + s[0]), e[1] * f1 - f2 * (q[1] + s[1]), -f2 * q[2],
                    -f2 * q[0], -f2 * q[1]};

    float c = SQ(f1) - (1.0f - SQ(s[0]) - SQ(s[1])) * SQ(q[2]);
    float dc[5] = {-2.0f * f1 * (q[0] + s[0]), -2.0f * f1 * (q[1] + s[1]),
                    -2.0f * f1 * q[2] - 2.0f * (1.0f - SQ(s[0]) - SQ(s[1])) * q[2],
                    -2.0f * f1 * q[0] + 2.0f * s[0] * SQ(q[2]), -2.0f * f1 * q[1] + 2.0f * s[1] * SQ(q[2])};
    float delta = sqrtf(SQ(b) - a * c);

    for (int i = 0; i < 5; i++)
    {
        qs_ctg[i] = t_ctg * ((db[i] + sign * 0.5f / delta * (2.0f * b * db[i] - da[i] * c - a * dc[i])) / a - (b + sign * delta) / SQ(a) * da[i]);
    }
}

int find_kins(float *out, unsigned char *mask, size_t N, int *hkl, unsigned *hidxs, float *basis, unsigned *bidxs,
              float *pupil, unsigned threads)
{
    /* Check parameters */
    if (!out || !mask || !hkl || !hidxs || !basis || !bidxs || !pupil)
    {ERROR("find_kins: one of the arguments is NULL."); return -1;}

    if (N == 0) return 0;

    /* Rectangular bounds consist of four lines defined as: r = s + t * e, t \el [lim[0], lim[1]] */
    float earr[8] = {0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f};
    float sarr[8] = {pupil[0], 0.0f, 0.0f, pupil[1], 0.0f, pupil[3], pupil[2], 0.0f};
    float lims[8] = {earr[0] * pupil[0] + earr[1] * pupil[1], earr[0] * pupil[2] + earr[1] * pupil[3],
                     earr[2] * pupil[0] + earr[3] * pupil[1], earr[2] * pupil[2] + earr[3] * pupil[3],
                     earr[4] * pupil[0] + earr[5] * pupil[1], earr[4] * pupil[2] + earr[5] * pupil[3],
                     earr[6] * pupil[0] + earr[7] * pupil[1], earr[6] * pupil[2] + earr[7] * pupil[3]};
    float NA = sqrtf(SQ(pupil[2]) + SQ(pupil[3]));

    #pragma omp parallel num_threads(threads)
    {
        int i, j, idx;
        float q_sq, q_rho, t, q[3];

        #pragma omp for
        for (int n = 0; n < (int)N; n++)
        {
            q[0] = hkl[3 * hidxs[n] + 0] * basis[9 * bidxs[n] + 0]
                 + hkl[3 * hidxs[n] + 1] * basis[9 * bidxs[n] + 3]
                 + hkl[3 * hidxs[n] + 2] * basis[9 * bidxs[n] + 6];
            q[1] = hkl[3 * hidxs[n] + 0] * basis[9 * bidxs[n] + 1]
                 + hkl[3 * hidxs[n] + 1] * basis[9 * bidxs[n] + 4]
                 + hkl[3 * hidxs[n] + 2] * basis[9 * bidxs[n] + 7];
            q[2] = hkl[3 * hidxs[n] + 0] * basis[9 * bidxs[n] + 2]
                 + hkl[3 * hidxs[n] + 1] * basis[9 * bidxs[n] + 5]
                 + hkl[3 * hidxs[n] + 2] * basis[9 * bidxs[n] + 8];

            q_sq = SQ(q[0]) + SQ(q[1]) + SQ(q[2]);
            q_rho = q[2] * sqrtf(1.0f / q_sq - 0.25) + 0.5f * sqrtf(SQ(q[0]) + SQ(q[1]));
            if ((q_sq < 4.0) && (fabsf(q_rho) < NA))
            {
                i = j = 0;
                while ((j < 2) && (i < 4))
                {
                    if (find_intersection(&t, q, earr + 2 * i, sarr + 2 * i, lims + 2 * i))
                    {
                        idx = 6 * n + 3 * j;
                        out[idx + 0] = sarr[2 * i + 0] + t * earr[2 * i + 0];
                        out[idx + 1] = sarr[2 * i + 1] + t * earr[2 * i + 1];
                        out[idx + 2] = sqrtf(1.0f - SQ(out[idx]) - SQ(out[idx + 1]));

                        j++;
                    }
                    i++;
                }

                if (j == 2) mask[n] = 1;
                else
                {
                    mask[n] = 0; memset(out + 6 * n, 0, 6 * sizeof(float));
                }
            }
            else
            {
                mask[n] = 0; memset(out + 6 * n, 0, 6 * sizeof(float));
            }
        }
    }

    return 0;
}

int find_kins_vjp(float *bout, float *kout, float *vec, size_t N, int *hkl, unsigned *hidxs, float *basis,
                  size_t bsize, unsigned *bidxs, float *pupil, unsigned threads)
{
    /* Check parameters */
    if (!bout || !kout || !vec || !hkl || !hidxs || !basis || !bidxs || !pupil)
    {ERROR("find_kins: one of the arguments is NULL."); return -1;}

    if (N == 0) return 0;

    /* Rectangular bounds consist of four lines defined as: r = s + t * e, t \el [lim[0], lim[1]] */
    float earr[8] = {0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f};
    float sarr[8] = {pupil[0], 0.0f, 0.0f, pupil[1], 0.0f, pupil[3], pupil[2], 0.0f};
    int kidx[8] = {0, -1, -1, 1, -1, 3, 2, -1};
    float lims[8] = {earr[0] * pupil[0] + earr[1] * pupil[1], earr[0] * pupil[2] + earr[1] * pupil[3],
                      earr[2] * pupil[0] + earr[3] * pupil[1], earr[2] * pupil[2] + earr[3] * pupil[3],
                      earr[4] * pupil[0] + earr[5] * pupil[1], earr[4] * pupil[2] + earr[5] * pupil[3],
                      earr[6] * pupil[0] + earr[7] * pupil[1], earr[6] * pupil[2] + earr[7] * pupil[3]};
    float NA = sqrtf(SQ(pupil[2]) + SQ(pupil[3]));

    #pragma omp parallel num_threads(threads)
    {
        int i, j, k, idx, sign;
        float q_sq, q_rho, t, t_ctg, q[3], kin[3], qs_ctg[5], b_ctg[9], k_ctg[4];

        float *bbuf = (float *)calloc(9 * bsize, sizeof(float));
        float kbuf[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        #pragma omp for
        for (int n = 0; n < (int)N; n++)
        {
            q[0] = hkl[3 * hidxs[n] + 0] * basis[9 * bidxs[n] + 0]
                 + hkl[3 * hidxs[n] + 1] * basis[9 * bidxs[n] + 3]
                 + hkl[3 * hidxs[n] + 2] * basis[9 * bidxs[n] + 6];
            q[1] = hkl[3 * hidxs[n] + 0] * basis[9 * bidxs[n] + 1]
                 + hkl[3 * hidxs[n] + 1] * basis[9 * bidxs[n] + 4]
                 + hkl[3 * hidxs[n] + 2] * basis[9 * bidxs[n] + 7];
            q[2] = hkl[3 * hidxs[n] + 0] * basis[9 * bidxs[n] + 2]
                 + hkl[3 * hidxs[n] + 1] * basis[9 * bidxs[n] + 5]
                 + hkl[3 * hidxs[n] + 2] * basis[9 * bidxs[n] + 8];

            q_sq = SQ(q[0]) + SQ(q[1]) + SQ(q[2]);
            q_rho = q[2] * sqrtf(1.0f / q_sq - 0.25) + 0.5f * sqrtf(SQ(q[0]) + SQ(q[1]));
            if ((q_sq < 4.0) && (fabsf(q_rho) < NA))
            {
                i = j = 0; 
                for (k = 0; k < 9; k++) b_ctg[k] = 0.0f;
                for (k = 0; k < 4; k++) k_ctg[k] = 0.0f;

                while ((j < 2) && (i < 4))
                {
                    sign = find_intersection(&t, q, earr + 2 * i, sarr + 2 * i, lims + 2 * i);

                    if (sign)
                    {
                        idx = 6 * n + 3 * j;
                        kin[0] = sarr[2 * i + 0] + t * earr[2 * i + 0];
                        kin[1] = sarr[2 * i + 1] + t * earr[2 * i + 1];
                        kin[2] = sqrtf(1.0f - SQ(kin[0]) - SQ(kin[1]));

                        t_ctg = vec[idx] * earr[2 * i] + vec[idx + 1] * earr[2 * i + 1]
                              - vec[idx + 2] * (kin[0] * earr[2 * i] + kin[1] * earr[2 * i + 1]) / kin[2];

                        find_intersection_vjp(qs_ctg, sign, t_ctg, q, earr + 2 * i, sarr + 2 * i, lims + 2 * i);

                        for (k = 0; k < 9; k++) b_ctg[k] += qs_ctg[k % 3] * hkl[3 * hidxs[n] + k / 3];
                        for (k = 0; k < 2; k++) if (kidx[2 * i + k] >= 0) k_ctg[kidx[2 * i + k]] += qs_ctg[3 + k];

                        j++;
                    }
                    i++;
                }

                if (j == 2)
                {
                    for (k = 0; k < 9; k++) bbuf[bidxs[n] + k] += b_ctg[k];
                    for (k = 0; k < 4; k++) kbuf[k] += k_ctg[k];
                }

            }
        }

        for (i = 0; i < (int)(9 * bsize); i++)
        {
            #pragma omp atomic
            bout[i] += bbuf[i];
        }
        for (i = 0; i < 4; i++)
        {
            #pragma omp atomic
            kout[i] += kbuf[i];
        }

        DEALLOC(bbuf);
    }

    return 0;
}