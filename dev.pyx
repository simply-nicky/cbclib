#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
cimport numpy as np
import numpy as np
import cython
from math import ceil
from libc.stdlib cimport free, malloc, calloc
from libc.string cimport memcmp
from cpython.ref cimport Py_INCREF
from cython.parallel import prange
from cbclib.bin.image_proc cimport check_array, normalize_sequence
from cbclib.bin.line_detector cimport ArrayWrapper

# Numpy must be initialized. When using numpy from C or Cython you must
# *ALWAYS* do that, or you will have segfaults
np.import_array()

DEF N_BINS = 1024
DEF LINE_SIZE = 7
DEF NOT_DEF = -1.0
DEF STACKSIZE = 1000000000 # If the input array is larger, you'll get an overflow segfault

def draw_lines_aa(np.ndarray image not None, np.ndarray lines not None, int max_val=255, int dilation=0) -> np.ndarray:
    """Draw thick lines with variable thickness and the antialiasing applied.
    The lines must follow the LSD convention, see the parameters for more info.

    Args:
        image (numpy.ndarray) : Image array.
        lines (numpy.ndarray) : An array of the detected lines. Must have a shape
            of (`N`, 7), where `N` is the number of lines. Each line is comprised
            of 7 parameters as follows:

            * `[x1, y1]`, `[x2, y2]` : The coordinates of the line's
            ends.
            * `width` : Line's width.
            * `p` : Angle precision [0, 1] given by angle tolerance
            over 180 degree.
            * `-log10(NFA)` : Number of false alarms.

        max_val (int) : Maximum value of the line mask.
        dilation (int) : Size of the binary dilation applied to the output image.

    Returns:
        numpy.ndarray : Output image with the lines drawn.

    See Also:
        :class:`cbclib.bin.LSD` : Line Segment Detector.
    """
    image = check_array(image, np.NPY_UINT32)
    lines = check_array(lines, np.NPY_FLOAT64)

    if image.ndim != 2:
        raise ValueError("image array must be two-dimensional")
    if lines.ndim != 2 or lines.shape[1] != 7:
        raise ValueError(f"lines array has an incompatible shape")

    cdef unsigned int *_image = <unsigned int *>np.PyArray_DATA(image)
    cdef unsigned long _Y = image.shape[0]
    cdef unsigned long _X = image.shape[1]
    cdef double *_lines = <double *>np.PyArray_DATA(lines)
    cdef unsigned long _n_lines = lines.shape[0]

    with nogil:
        fail = draw_lines(_image, _Y, _X, max_val, _lines, _n_lines, dilation)
    if fail:
        raise RuntimeError('C backend exited with error.')    
    return image

def draw_line_indices_aa(np.ndarray lines not None, object shape not None, int max_val=255, int dilation=0) -> np.ndarray:
    """Draw thick lines with variable thickness and the antialiasing applied.
    The lines must follow the LSD convention, see the parameters for more info.

    Args:
        lines (numpy.ndarray) : An array of the detected lines. Must have a shape
            of (`N`, 7), where `N` is the number of lines. Each line is comprised
            of 7 parameters as follows:

            * `[x1, y1]`, `[x2, y2]` : The coordinates of the line's
            ends.
            * `width` : Line's width.
            * `p` : Angle precision [0, 1] given by angle tolerance
            over 180 degree.
            * `-log10(NFA)` : Number of false alarms.

        shape (Iterable[int]) : Shape of the image.
        max_val (int) : Maximum value of the line mask.
        dilation (int) : Size of the binary dilation applied to the output image.

    Returns:
        numpy.ndarray : Output line indices.

    See Also:
        :class:`cbclib.bin.LSD` : Line Segment Detector.
    """
    lines = check_array(lines, np.NPY_FLOAT64)

    if lines.ndim != 2 or lines.shape[1] != 7:
        raise ValueError(f"lines array has an incompatible shape")

    cdef np.ndarray _shape = normalize_sequence(shape, 2, np.NPY_INTP)
    cdef unsigned long _Y = _shape[0]
    cdef unsigned long _X = _shape[1]

    cdef unsigned int *_idxs
    cdef unsigned long _n_idxs
    cdef double *_lines = <double *>np.PyArray_DATA(lines)
    cdef unsigned long _n_lines = lines.shape[0]

    with nogil:
        fail = draw_line_indices(&_idxs, &_n_idxs, _Y, _X, max_val, _lines, _n_lines, dilation)
    if fail:
        raise RuntimeError('C backend exited with error.')    

    cdef np.npy_intp *dim = [_n_idxs, 4]
    cdef np.ndarray idxs = ArrayWrapper.from_ptr(<void *>_idxs).to_ndarray(2, dim, np.NPY_UINT32)

    return idxs

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