cdef extern from "img_proc.h":
    int compute_euler_angles(double *eulers, double *rot_mats, unsigned long n_mats) nogil
    int compute_euler_matrix(double *rot_mats, double *eulers, unsigned long n_mats) nogil
    int compute_tilt_matrix(double *rot_mats, double *angles, unsigned long n_mats, 
                            double a0, double a1, double a2) nogil