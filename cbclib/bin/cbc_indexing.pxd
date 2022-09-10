cdef extern from "img_proc.h":
    int compute_euler_angles(double *angles, double *rot_mats, unsigned long n_mats) nogil
    int compute_euler_matrix(double *rot_mats, double *angles, unsigned long n_mats) nogil
    int compute_tilt_angles(double *angles, double *rot_mats, unsigned long n_mats) nogil
    int compute_tilt_matrix(double *rot_mats, double *angles, unsigned long n_mats) nogil
    int compute_rotations(double *rot_mats, double *as, double *bs, unsigned long n_mats) nogil