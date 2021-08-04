#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
cimport numpy as np
import numpy as np
import cython
from libc.stdlib cimport free, malloc, calloc
from libc.string cimport memcmp
from cpython.ref cimport Py_INCREF

# Numpy must be initialized. When using numpy from C or Cython you must
# *ALWAYS* do that, or you will have segfaults
np.import_array()

DEF QUANT = 2.0
DEF ANG_TH = 45.
DEF DENSITY_TH = 0.7
DEF N_BINS = 1024
DEF LINE_SIZE = 7

cdef class ArrayWrapper:
    """A wrapper class for a C data structure. """
    cdef void* _data

    def __cinit__(self):
        self._data = NULL

    def __dealloc__(self):
        if not self._data is NULL:
            free(self._data)
            self._data = NULL

    @staticmethod
    cdef ArrayWrapper from_ptr(void *data):
        """Factory function to create a new wrapper from a C pointer."""
        cdef ArrayWrapper wrapper = ArrayWrapper.__new__(ArrayWrapper)
        wrapper._data = data
        return wrapper

    cdef np.ndarray to_ndarray(self, int ndim, np.npy_intp *dims, int type_num):
        """Get a NumPy array from a wrapper."""
        cdef np.ndarray ndarray = np.PyArray_SimpleNewFromData(ndim, dims, type_num, self._data)

        # without this, data would be cleaned up right away
        Py_INCREF(self)
        np.PyArray_SetBaseObject(ndarray, self)
        return ndarray

cdef class LSD:
    cdef double _scale
    cdef double _sigma_scale
    cdef double _log_eps

    def __cinit__(self, double scale=0.8, double sigma_scale=0.6, double log_eps=0.):
        self._scale = scale
        self._sigma_scale = sigma_scale
        self._log_eps = log_eps

    def __init__(self, scale=0.8, sigma_scale=0.6, log_eps=0):
        """Line Segment Detector.
        """

    @staticmethod
    cdef np.ndarray _check_image(np.ndarray image):
        cdef int ndim = image.ndim
        if ndim != 2:
            raise ValueError('Image must be a 2D array.')
        if not np.PyArray_IS_C_CONTIGUOUS(image):
            image = np.PyArray_GETCONTIGUOUS(image)
        cdef int tn = np.PyArray_TYPE(image)
        if tn != np.NPY_FLOAT64:
            image = np.PyArray_Cast(image, np.NPY_FLOAT64)
        return image

    cpdef dict detect(self, np.ndarray image):
        image = LSD._check_image(image)

        cdef double *_img = <double *>np.PyArray_DATA(image)
        cdef int _img_x = <int>image.shape[1]
        cdef int _img_y = <int>image.shape[0]

        cdef int _reg_x
        cdef int _reg_y
        cdef int *_reg_img

        cdef int _n_out
        cdef double *_out

        cdef int fail = 0
        with nogil:
            fail =  LineSegmentDetection(&_out, &_n_out, _img, _img_x, _img_y,
                                         self._scale, self._sigma_scale, QUANT,
                                         ANG_TH, self._log_eps, DENSITY_TH, N_BINS,
                                         &_reg_img, &_reg_x, &_reg_y)

        if fail:
            raise RuntimeError("LSD execution finished with an error.")
        
        cdef np.npy_intp *out_dims = [_n_out, LINE_SIZE]
        cdef np.ndarray out = ArrayWrapper.from_ptr(<void *>_out).to_ndarray(2, out_dims, np.NPY_FLOAT64)

        cdef np.npy_intp *reg_dims = [_reg_y, _reg_x,]
        cdef np.ndarray reg_img = ArrayWrapper.from_ptr(<void *>_reg_img).to_ndarray(2, reg_dims, np.NPY_INT32)

        return {'lines': out, 'labels': reg_img}

cdef int extend_mode_to_code(str mode) except -1:
    if mode == 'constant':
        return EXTEND_CONSTANT
    elif mode == 'nearest':
        return EXTEND_NEAREST
    elif mode == 'mirror':
        return EXTEND_MIRROR
    elif mode == 'reflect':
        return EXTEND_REFLECT
    elif mode == 'wrap':
        return EXTEND_WRAP
    else:
        raise RuntimeError('boundary mode not supported')

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

# def median_filter(data: np.ndarray, mask: np.ndarray, size: cython.uint=3, axis: cython.int=0,
#                   mode: str='reflect', cval: cython.double=0., num_threads: cython.uint=1) -> np.ndarray:
#     """Calculate a median along the `axis`.

#     Parameters
#     ----------
#     data : numpy.ndarray
#         Intensity frames.
#     mask : numpy.ndarray
#         Bad pixel mask.
#     size : int, optional
#         `size` gives the shape that is taken from the input array, at every element position,
#         to define the input to the filter function. Default is 3.
#     axis : int, optional
#         Array axis along which median values are calculated.
#     mode : {'constant', 'nearest', 'mirror', 'reflect', 'wrap'}, optional
#         The mode parameter determines how the input array is extended when the filter
#         overlaps a border. Default value is 'reflect'. The valid values and their behavior
#         is as follows:

#         * 'constant', (k k k k | a b c d | k k k k) : The input is extended by filling all
#           values beyond the edge with the same constant value, defined by the `cval`
#           parameter.
#         * 'nearest', (a a a a | a b c d | d d d d) : The input is extended by replicating
#           the last pixel.
#         * 'mirror', (c d c b | a b c d | c b a b) : The input is extended by reflecting
#           about the center of the last pixel. This mode is also sometimes referred to as
#           whole-sample symmetric.
#         * 'reflect', (d c b a | a b c d | d c b a) : The input is extended by reflecting
#           about the edge of the last pixel. This mode is also sometimes referred to as
#           half-sample symmetric.
#         * 'wrap', (a b c d | a b c d | a b c d) : The input is extended by wrapping around
#           to the opposite edge.
#     cval : float, optional
#         Value to fill past edges of input if mode is ‘constant’. Default is 0.0.
#     num_threads : int, optional
#         Number of threads.

#     Returns
#     -------
#     wfield : numpy.ndarray
#         Whitefield.
#     """
#     data = check_array(data, np.NPY_FLOAT64)
#     mask = check_array(mask, np.NPY_BOOL)

#     cdef int ndim = data.ndim
#     if memcmp(data.shape, mask.shape, ndim * sizeof(np.npy_intp)):
#         raise ValueError('mask and data arrays must have identical shapes')
#     axis = axis if axis >= 0 else ndim + axis
#     axis = axis if axis <= ndim - 1 else ndim - 1
#     cdef np.npy_intp *dims = data.shape
#     cdef unsigned long *_dims = <unsigned long *>dims
#     cdef int type_num = np.PyArray_TYPE(data)
#     cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(ndim, dims, type_num)
#     cdef void *_out = <void *>np.PyArray_DATA(out)
#     cdef void *_data = <void *>np.PyArray_DATA(data)
#     cdef unsigned char *_mask = <unsigned char *>np.PyArray_DATA(mask)
#     cdef int _mode = extend_mode_to_code(mode)
#     cdef void *_cval = <void *>&cval
#     with nogil:
#         if type_num == np.NPY_FLOAT64:
#             fail = median_filter_c(_out, _data, _mask, ndim, _dims, 8, axis, size, _mode, _cval, compare_double, num_threads)
#         elif type_num == np.NPY_FLOAT32:
#             fail = median_filter_c(_out, _data, _mask, ndim, _dims, 4, axis, size, _mode, _cval, compare_float, num_threads)
#         elif type_num == np.NPY_INT32:
#             fail = median_filter_c(_out, _data, _mask, ndim, _dims, 4, axis, size, _mode, _cval, compare_long, num_threads)
#         else:
#             raise TypeError('data argument has incompatible type: {:s}'.format(data.dtype))
#     return out

def median_filter(data: np.ndarray, footprint: object, axis: cython.int=0,
                  mode: str='reflect', cval: cython.double=0., num_threads: cython.uint=1) -> np.ndarray:
    """Calculate a median along the `axis`.

    Parameters
    ----------
    data : numpy.ndarray
        Intensity frames.
    mask : numpy.ndarray
        Bad pixel mask.
    size : int, optional
        `size` gives the shape that is taken from the input array, at every element position,
        to define the input to the filter function. Default is 3.
    axis : int, optional
        Array axis along which median values are calculated.
    mode : {'constant', 'nearest', 'mirror', 'reflect', 'wrap'}, optional
        The mode parameter determines how the input array is extended when the filter
        overlaps a border. Default value is 'reflect'. The valid values and their behavior
        is as follows:

        * 'constant', (k k k k | a b c d | k k k k) : The input is extended by filling all
          values beyond the edge with the same constant value, defined by the `cval`
          parameter.
        * 'nearest', (a a a a | a b c d | d d d d) : The input is extended by replicating
          the last pixel.
        * 'mirror', (c d c b | a b c d | c b a b) : The input is extended by reflecting
          about the center of the last pixel. This mode is also sometimes referred to as
          whole-sample symmetric.
        * 'reflect', (d c b a | a b c d | d c b a) : The input is extended by reflecting
          about the edge of the last pixel. This mode is also sometimes referred to as
          half-sample symmetric.
        * 'wrap', (a b c d | a b c d | a b c d) : The input is extended by wrapping around
          to the opposite edge.
    cval : float, optional
        Value to fill past edges of input if mode is ‘constant’. Default is 0.0.
    num_threads : int, optional
        Number of threads.

    Returns
    -------
    wfield : numpy.ndarray
        Whitefield.
    """
    data = check_array(data, np.NPY_FLOAT64)

    cdef int ndim = data.ndim
    cdef np.ndarray fsize = normalize_sequence(footprint, ndim, np.NPY_INTP)
    cdef unsigned long *_fsize = <unsigned long *>np.PyArray_DATA(fsize)

    cdef np.npy_intp *dims = data.shape
    cdef unsigned long *_dims = <unsigned long *>dims
    cdef int type_num = np.PyArray_TYPE(data)
    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(ndim, dims, type_num)
    cdef void *_out = <void *>np.PyArray_DATA(out)
    cdef void *_data = <void *>np.PyArray_DATA(data)
    cdef int _mode = extend_mode_to_code(mode)
    cdef void *_cval = <void *>&cval
    with nogil:
        if type_num == np.NPY_FLOAT64:
            fail = median_filter_c(_out, _data, ndim, _dims, 8, _fsize, _mode, _cval, compare_double, num_threads)
        elif type_num == np.NPY_FLOAT32:
            fail = median_filter_c(_out, _data, ndim, _dims, 4, _fsize, _mode, _cval, compare_float, num_threads)
        elif type_num == np.NPY_INT32:
            fail = median_filter_c(_out, _data, ndim, _dims, 4, _fsize, _mode, _cval, compare_long, num_threads)
        else:
            raise TypeError('data argument has incompatible type: {:s}'.format(data.dtype))
    return out

# def test():
#     cdef void *buffer = calloc(10, sizeof(double))
#     if buffer is NULL:
#         raise MemoryError()

#     cdef ArrayWrapper wrapper = ArrayWrapper.from_ptr(buffer)
#     cdef np.npy_intp *dims = [10,]
#     cdef np.ndarray arr = wrapper.to_ndarray(1, dims, np.NPY_FLOAT64)

#     return arr
