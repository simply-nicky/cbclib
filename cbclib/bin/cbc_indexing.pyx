cimport numpy as np
import numpy as np
import cython

# Numpy must be initialized. When using numpy from C or Cython you must
# *ALWAYS* do that, or you will have segfaults
np.import_array()

cdef np.ndarray check_array(np.ndarray array, int type_num):
    if not np.PyArray_IS_C_CONTIGUOUS(array):
        array = np.PyArray_GETCONTIGUOUS(array)
    cdef int tn = np.PyArray_TYPE(array)
    if tn != type_num:
        array = np.PyArray_Cast(array, type_num)
    return array

cdef np.ndarray number_to_array(object num, np.npy_intp rank, int type_num):
    cdef np.npy_intp *dims = [rank,]
    cdef np.ndarray arr = <np.ndarray>np.PyArray_SimpleNew(1, dims, type_num)
    cdef int i
    for i in range(rank):
        arr[i] = num
    return arr

cdef np.ndarray normalize_sequence(object inp, np.npy_intp rank, int type_num):
    # If input is a scalar, create a sequence of length equal to the
    # rank by duplicating the input. If input is a sequence,
    # check if its length is equal to the length of array.
    cdef np.ndarray arr
    cdef int tn
    if np.PyArray_IsAnyScalar(inp):
        arr = number_to_array(inp, rank, type_num)
    elif np.PyArray_Check(inp):
        arr = <np.ndarray>inp
        tn = np.PyArray_TYPE(arr)
        if tn != type_num:
            arr = <np.ndarray>np.PyArray_Cast(arr, type_num)
    elif isinstance(inp, (list, tuple)):
        arr = <np.ndarray>np.PyArray_FROM_OTF(inp, type_num, np.NPY_ARRAY_C_CONTIGUOUS)
    else:
        raise ValueError("Wrong sequence argument type")
    cdef np.npy_intp size = np.PyArray_SIZE(arr)
    if size != rank:
        raise ValueError("Sequence argument must have length equal to input rank")
    return arr

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