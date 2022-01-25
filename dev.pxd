#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
cdef extern from "img_proc.h":
    int draw_lines (unsigned int *out, unsigned long Y, unsigned long X, unsigned int max_val,
                    double *lines, unsigned long lines, unsigned int dilation) nogil

    int draw_line_indices(unsigned int **out, unsigned long *n_idxs, unsigned long Y, unsigned long X,
                          unsigned int max_val, double *lines, unsigned long n_lines,
                          unsigned int dilation) nogil

    int generate_rot_matrix(double *rot_mats, double *angles, unsigned long n_mats, 
                            double a0, double a1, double a2) nogil

cdef enum:
    EXTEND_CONSTANT = 0
    EXTEND_NEAREST = 1
    EXTEND_MIRROR = 2
    EXTEND_REFLECT = 3
    EXTEND_WRAP = 4