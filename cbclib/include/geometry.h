#ifndef GEOMETRY_H
#define GEOMETRY_H
#include "include.h"

int compute_euler_angles(double *angles, double *rot_mats, size_t n_mats);
int compute_euler_matrix(double *rot_mats, double *angles, size_t n_mats);
int compute_tilt_angles(double *angles, double *rot_mats, size_t n_mats);
int compute_tilt_matrix(double *rot_mats, double *angles, size_t n_mats);
int compute_rotations(double *rot_mats, double *as, double *bs, size_t n_mats);

int det2k(double *karr, double *x, double *y, unsigned *idxs, size_t ksize, double *src, unsigned threads);
int k2det(double *x, double *y, double *karr, unsigned *idxs, size_t ksize, double *src, unsigned threads);
int k2smp(double *pts, double *karr, unsigned *idxs, size_t ksize, double *z, double *src, unsigned threads);
int rotate_vec(double *out, double *vecs, unsigned *idxs, size_t vsize, double *rmats, unsigned threads);

#endif