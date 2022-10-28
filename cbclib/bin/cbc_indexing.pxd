cdef enum:
    SEARCH_LEFT = 0
    SEARCH_RIGHT = 1

cdef inline int side_to_code(str side) except -1:
    if side == 'left':
        return SEARCH_LEFT
    elif side == 'right':
        return SEARCH_RIGHT
    else:
        raise RuntimeError(f'Invalid side keyword: {side}')

cdef extern from "array.h":
    int compare_double(void *a, void *b) nogil
    int compare_float(void *a, void *b) nogil
    int compare_int(void *a, void *b) nogil
    int compare_uint(void *a, void *b) nogil
    int compare_ulong(void *a, void *b) nogil

    unsigned long searchsorted_c "searchsorted" (void *key, void *base, unsigned long npts,
                                 unsigned long size, int side, int (*compar)(void*, void*)) nogil

cdef extern from "img_proc.h":
    int compute_euler_angles(double *angles, double *rot_mats, unsigned long n_mats) nogil
    int compute_euler_matrix(double *rot_mats, double *angles, unsigned long n_mats) nogil
    int compute_tilt_angles(double *angles, double *rot_mats, unsigned long n_mats) nogil
    int compute_tilt_matrix(double *rot_mats, double *angles, unsigned long n_mats) nogil
    int compute_rotations(double *rot_mats, double *as, double *bs, unsigned long n_mats) nogil