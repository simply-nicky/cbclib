cimport numpy as np
import numpy as np
import cython
from .image_proc cimport check_array, normalize_sequence

# Numpy must be initialized. When using numpy from C or Cython you must
# *ALWAYS* do that, or you will have segfaults
np.import_array()

def euler_angles(rot_mats: np.ndarray) -> np.ndarray:
    rot_mats = check_array(rot_mats, np.NPY_FLOAT64)

    cdef np.npy_intp *edims = [rot_mats.shape[0], 3]
    cdef np.ndarray eulers = <np.ndarray>np.PyArray_SimpleNew(2, edims, np.NPY_FLOAT64)

    cdef double *e_ptr = <double *>np.PyArray_DATA(eulers)
    cdef double *rm_ptr = <double *>np.PyArray_DATA(rot_mats)
    cdef unsigned long n_mats = rot_mats.shape[0]

    cdef int fail = 0
    with nogil:
        fail = compute_euler_angles(e_ptr, rm_ptr, n_mats)
    if fail:
        raise RuntimeError('C backend exited with error.')
    return eulers

def euler_matrix(eulers: np.ndarray) -> np.ndarray:
    eulers = check_array(eulers, np.NPY_FLOAT64)

    cdef np.npy_intp *rmdims = [eulers.shape[0], 3, 3]
    cdef np.ndarray rot_mats = <np.ndarray>np.PyArray_SimpleNew(3, rmdims, np.NPY_FLOAT64)

    cdef double *e_ptr = <double *>np.PyArray_DATA(eulers)
    cdef double *rm_ptr = <double *>np.PyArray_DATA(rot_mats)
    cdef unsigned long n_mats = eulers.shape[0]

    cdef int fail = 0
    with nogil:
        fail = compute_rot_matrix(rm_ptr, e_ptr, n_mats)
    if fail:
        raise RuntimeError('C backend exited with error.')
    return rot_mats

def tilt_matrix(tilts: np.ndarray, axis: object) -> np.ndarray:
    cdef np.ndarray _axis = normalize_sequence(axis, 3, np.NPY_FLOAT64)

    cdef np.npy_intp *rmdims = [tilts.shape[0], 3, 3]
    cdef np.ndarray rot_mats = <np.ndarray>np.PyArray_SimpleNew(3, rmdims, np.NPY_FLOAT64)

    cdef double *t_ptr = <double *>np.PyArray_DATA(tilts)
    cdef double *rm_ptr = <double *>np.PyArray_DATA(rot_mats)
    cdef unsigned long n_mats = tilts.shape[0]
    cdef double a0 = _axis[0], a1 = _axis[1], a2 = _axis[2]

    cdef int fail = 0
    with nogil:
        fail = generate_rot_matrix(rm_ptr, t_ptr, n_mats, a0, a1, a2)
    if fail:
        raise RuntimeError('C backend exited with error.')
    return rot_mats