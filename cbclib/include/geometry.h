#ifndef GEOMETRY_H
#define GEOMETRY_H
#include "include.h"

/*----------------------------------------------------------------------------*/
/*--------------------- Rotations with Euler angles --------------------------*/
/*----------------------------------------------------------------------------*/
int compute_euler_angles(float *angles, float *rot_mats, size_t n_mats);
int compute_euler_matrix(float *rot_mats, float *angles, size_t n_mats);
int compute_tilt_angles(float *angles, float *rot_mats, size_t n_mats);
int compute_tilt_matrix(float *rot_mats, float *angles, size_t n_mats);
int compute_rotations(float *rot_mats, float *as, float *bs, size_t n_mats);

int rotate_vec(float *out, float *vecs, unsigned *idxs, size_t vsize, float *rmats, unsigned threads);

/*----------------------------------------------------------------------------*/
/*---------- Conversions from the real space to the reciprocal ---------------*/
/*----------------------------------------------------------------------------*/
int det2k(float *karr, float *x, float *y, unsigned *idxs, size_t ksize, float *src, unsigned threads);
int det2k_vjp(float *xout, float *yout, float *sout, float *vec, float *x, float *y, unsigned *idxs,
              size_t ksize, float *src, size_t ssize, unsigned threads);

int k2det(float *x, float *y, float *karr, unsigned *idxs, size_t ksize, float *src, unsigned threads);
int k2det_vjp(float *kout, float *sout, float *xvec, float *yvec, float *karr, unsigned *idxs, size_t ksize,
              float *src, size_t ssize, unsigned threads);

int k2smp(float *pts, float *karr, unsigned *idxs, size_t ksize, float *z, float *src, unsigned threads);
int k2smp_vjp(float *kout, float *zout, float *sout, float *vec, float *karr, unsigned *idxs, size_t ksize,
              float *z, size_t zsize, float *src, unsigned threads);

/*----------------------------------------------------------------------------*/
/*------------------ Finding source lines for CBD model ----------------------*/
/*----------------------------------------------------------------------------*/
int find_kins(float *out, unsigned char *mask, size_t N, int *hkl, unsigned *hidxs, float *basis, unsigned *bidxs,
              float *pupil, unsigned threads);
int find_kins_vjp(float *bout, float *kout, float *vec, size_t N, int *hkl, unsigned *hidxs, float *basis,
                  size_t bsize, unsigned *bidxs, float *pupil, unsigned threads);

#endif