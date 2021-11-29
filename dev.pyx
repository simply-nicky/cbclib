#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
cimport numpy as np
import numpy as np
import cython
from math import ceil
from libc.stdlib cimport free, malloc, calloc
from libc.string cimport memcmp
from cpython.ref cimport Py_INCREF
from cython.parallel import prange

# Numpy must be initialized. When using numpy from C or Cython you must
# *ALWAYS* do that, or you will have segfaults
np.import_array()

DEF N_BINS = 1024
DEF LINE_SIZE = 7
DEF NOT_DEF = -1.0
DEF STACKSIZE = 1000000000 # If the input array is larger, you'll get an overflow segfault

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
    """LSD  is a class for performing the streak detection
    on digital images with Line Segment Detector algorithm [LSD]_.

    References
    ----------
    .. [LSD] "LSD: a Line Segment Detector" by Rafael Grompone von Gioi,
             Jeremie Jakubowicz, Jean-Michel Morel, and Gregory Randall,
             Image Processing On Line, 2012. DOI:10.5201/ipol.2012.gjmr-lsd
             http://dx.doi.org/10.5201/ipol.2012.gjmr-lsd
    """
    cdef public double ang_th
    cdef public double density_th
    cdef public double log_eps
    cdef public double scale
    cdef public double sigma_scale
    cdef public double quant
    cdef public double x_c
    cdef public double y_c

    def __cinit__(self, double scale=0.8, double sigma_scale=0.6, double log_eps=0.,
                  double ang_th=45.0, double density_th=0.7, double quant=2.0,
                  double x_c=NOT_DEF, double y_c=NOT_DEF):
        if scale < 0 or scale > 1:
            raise ValueError('scale is out of bounds (0.0, 1.0)')
        else:
            self.scale = scale
        if sigma_scale < 0 or sigma_scale > 1:
            raise ValueError('sigma_scale is out of bounds (0.0, 1.0)')
        else:
            self.sigma_scale = sigma_scale
        self.log_eps = log_eps
        if ang_th < 0 or ang_th > 360:
            raise ValueError('ang_th is out of bounds (0.0, 360.0)')
        else:
            self.ang_th = ang_th
        if density_th < 0 or density_th > 1:
            raise ValueError('density_th is out of bounds (0.0, 1.0)')
        else:
            self.density_th = density_th
        if quant < 0:
            raise ValueError('quant msut be positive')
        else:
            self.quant = quant
        self.x_c = x_c
        self.y_c = y_c

    def __init__(self, scale=0.8, sigma_scale=0.6, log_eps=0, ang_th=45.0,
                 density_th=0.7, quant=2.0, double x_c=NOT_DEF, double y_c=NOT_DEF):
        """Create a LSD object for streak detection on digital images.

        Parameters
        ----------
        scale : float, optional
            When different from 1.0, LSD will scale the input image
            by 'scale' factor by Gaussian filtering, before detecting
            line segments. Default value is 0.8.
        sigma_scale : float, optional
            When `scale` is different from 1.0, the sigma of the Gaussian
            filter is :code:`sigma = sigma_scale / scale`, if scale is less
            than 1.0, and :code:`sigma = sigma_scale` otherwise. Default
            value is 0.6.
        log_eps : float, optional
            Detection threshold, accept if -log10(NFA) > log_eps.
            The larger the value, the more strict the detector is, and will
            result in less detections. The value -log10(NFA) is equivalent
            but more intuitive than NFA:

            * -1.0 gives an average of 10 false detections on noise.
            *  0.0 gives an average of 1 false detections on noise.
            *  1.0 gives an average of 0.1 false detections on nose.
            *  2.0 gives an average of 0.01 false detections on noise.
            Default value is 0.0.
        ang_th : float, optional
            Gradient angle tolerance in the region growing algorithm, in
            degrees. Default value is 45.0.
        density_th : float, optional
            Minimal proportion of 'supporting' points in a rectangle.
            Default value is 0.7.
        quant : float, optional
            Bound to the quantization error on the gradient norm.
            Example: if gray levels are quantized to integer steps,
            the gradient (computed by finite differences) error
            due to quantization will be bounded by 2.0, as the
            worst case is when the error are 1 and -1, that
            gives an error of 2.0. Default value is 2.0.
        """

    @staticmethod
    cdef np.ndarray _check_image(np.ndarray image):
        if not np.PyArray_IS_C_CONTIGUOUS(image):
            image = np.PyArray_GETCONTIGUOUS(image)
        cdef int tn = np.PyArray_TYPE(image)
        if tn != np.NPY_FLOAT64:
            image = np.PyArray_Cast(image, np.NPY_FLOAT64)
        return image

    cpdef dict detect(self, np.ndarray image, double radius=1.0,
                      bint filter_lines=False, bint return_labels=False,
                      unsigned int num_threads=1):
        """Perform the LSD streak detection on `image`.

        Parameters
        ----------
        image : np.ndarray
            2D array of the digital image.
        
        Returns
        -------
        dict
            :class:`dict` with the following fields:

            * `lines` : An array of the detected lines. Each line is
            comprised of 7 parameters as follows:

                * `[x1, y1]`, `[x2, y2]` : The coordinates of the line's
                  ends.
                * `width` : Line's width.
                * `p` : Angle precision [0, 1] given by angle tolerance
                  over 180 degree.
                * `-log10(NFA)` : Number of false alarms.
            
            * `labels` : image where each pixel indicates the line
              segment to which it belongs. Unused pixels have the value
              0, while the used ones have the number of the line segment,
              numbered in the same order as in `lines`.
        """
        if image.ndim < 2:
            raise ValueError('Image must be a 2D array.')
        image = LSD._check_image(image)

        cdef int ndim = image.ndim
        cdef double *_img = <double *>np.PyArray_DATA(image)
        cdef int _X = <int>image.shape[ndim - 1]
        cdef int _Y = <int>image.shape[ndim - 2]
        cdef int repeats = image.size / _X / _Y
        cdef np.ndarray streaks

        cdef double **_outs = <double **>malloc(repeats * sizeof(double *))
        if _outs is NULL:
            raise MemoryError('not enough memory')

        cdef int *_ns = <int *>malloc(repeats * sizeof(int))
        if _ns is NULL:
            free(_outs)
            raise MemoryError('not enough memory')

        cdef int **_regs = <int **>malloc(repeats * sizeof(int *))
        if _regs is NULL:
            free(_outs); free(_ns)
            raise MemoryError('not enough memory')

        cdef int *_reg_xs = <int *>malloc(repeats * sizeof(int))
        if _reg_xs is NULL:
            free(_outs); free(_ns); free(_regs)
            raise MemoryError('not enough memory')
        
        cdef int *_reg_ys = <int *>malloc(repeats * sizeof(int))
        if _reg_ys is NULL:
            free(_outs); free(_ns); free(_regs); free(_reg_xs)
            raise MemoryError('not enough memory')

        cdef int fail = 0, i
        cdef dict line_dict = {}, reg_dict = {}, out_dict = {}
        cdef np.npy_intp *out_dims = [0, LINE_SIZE]

        num_threads = repeats if <int>num_threads > repeats else <int>num_threads

        if filter_lines and self.x_c > 0.0 and self.x_c < _X and self.y_c > 0 and self.y_c < _Y:
            for i in prange(repeats, schedule='guided', num_threads=num_threads, nogil=True):
                fail |= LineSegmentDetection(&_outs[i], &_ns[i], _img + i * _Y * _X, _Y, _X,
                                            self.scale, self.sigma_scale, self.quant,
                                            self.ang_th, self.log_eps, self.density_th, N_BINS,
                                            &_regs[i], &_reg_ys[i], &_reg_xs[i])
                fail |= filter_lines_c(_outs[i], _img + i * _Y * _X, _Y, _X, _outs[i], _ns[i],
                                       self.x_c, self.y_c, radius)
        else:
            for i in prange(repeats, schedule='guided', num_threads=num_threads, nogil=True):
                fail |= LineSegmentDetection(&_outs[i], &_ns[i], _img + i * _Y * _X, _Y, _X,
                                            self.scale, self.sigma_scale, self.quant,
                                            self.ang_th, self.log_eps, self.density_th, N_BINS,
                                            &_regs[i], &_reg_ys[i], &_reg_xs[i])

        if fail:
            raise RuntimeError("LSD execution finished with an error")

        for i in range(repeats):
            out_dims[0] = _ns[i]
            streaks = ArrayWrapper.from_ptr(<void *>_outs[i]).to_ndarray(2, out_dims, np.NPY_FLOAT64)
            line_dict[i] = np.PyArray_Compress(streaks, streaks[:, 0], 0, <np.ndarray>NULL)

        out_dict['lines'] = line_dict

        if return_labels:
            for i in range(repeats):
                out_dims[0] = _reg_ys[i]
                out_dims[1] = _reg_xs[i]
                reg_dict[i] = ArrayWrapper.from_ptr(<void *>_regs[i]).to_ndarray(2, out_dims, np.NPY_INT32)
            
            out_dict['labels'] = reg_dict
        else:
            for i in range(repeats):
                free(_regs[i])

        free(_outs); free(_ns); free(_regs); free(_reg_xs); free(_reg_ys)

        return out_dict

    cpdef dict mask(self, np.ndarray image, unsigned int max_val=1,
                    unsigned int dilation=0, double radius=1.0,
                    bint filter_lines=False, bint return_lines=True,
                    unsigned int num_threads=1):
        """Perform the streak detection on `image` and return rasterized lines
        drawn on a mask array.

        Parameters
        ----------
        image : np.ndarray
            2D array of the digital image.
        max_val : int, optional
            Maximal value in the output mask.
        dilation : int, optional
            Size of the morphology dilation applied to the output mask.
        num_threads : int, optional
            Number of the computational threads.
        
        Returns
        -------
        mask : np.ndarray
            Array, that has the same shape as `image`, with the regions
            masked by the detected lines.
        """
        if image.ndim < 2:
            raise ValueError('Image must be >=2D array.')
        image = LSD._check_image(image)

        cdef int ndim = image.ndim
        cdef double *_img = <double *>np.PyArray_DATA(image)
        cdef int _X = <int>image.shape[ndim - 1]
        cdef int _Y = <int>image.shape[ndim - 2]

        cdef np.ndarray streaks
        cdef np.ndarray mask = np.PyArray_ZEROS(ndim, image.shape, np.NPY_UINT32, 0)
        cdef unsigned int *msk_ptr = <unsigned int *>np.PyArray_DATA(mask)
        cdef int repeats = image.size / _X / _Y

        cdef double **_outs = <double **>malloc(repeats * sizeof(double *))
        if _outs is NULL:
            raise MemoryError('not enough memory')

        cdef int *_ns = <int *>malloc(repeats * sizeof(int))
        if _ns is NULL:
            free(_outs)
            raise MemoryError('not enough memory')

        cdef int fail = 0, i
        cdef dict line_dict = {}, out_dict = {}
        cdef np.npy_intp *out_dims = [0, LINE_SIZE]

        num_threads = repeats if <int>num_threads > repeats else <int>num_threads

        if filter_lines and self.x_c > 0.0 and self.x_c < _X and self.y_c > 0.0 and self.y_c < _Y:
            for i in prange(repeats, schedule='guided', num_threads=num_threads, nogil=True):
                fail |= LineSegmentDetection(&_outs[i], &_ns[i], _img + i * _Y * _X, _Y, _X,
                                             self.scale, self.sigma_scale, self.quant,
                                             self.ang_th, self.log_eps, self.density_th, N_BINS,
                                             NULL, NULL, NULL)
                fail |= filter_lines_c(_outs[i], _img + i * _Y * _X, _Y, _X, _outs[i], _ns[i],
                                       self.x_c, self.y_c, radius)
                draw_lines_c(msk_ptr + i * _Y * _X, _Y, _X, max_val, _outs[i], _ns[i], dilation)
        else:
            for i in prange(repeats, schedule='guided', num_threads=num_threads, nogil=True):
                fail |= LineSegmentDetection(&_outs[i], &_ns[i], _img + i * _Y * _X, _Y, _X,
                                             self.scale, self.sigma_scale, self.quant,
                                             self.ang_th, self.log_eps, self.density_th, N_BINS,
                                             NULL, NULL, NULL)
                draw_lines_c(msk_ptr + i * _Y * _X, _Y, _X, max_val, _outs[i], _ns[i], dilation)

        if fail:
            raise RuntimeError("LSD execution finished with an error")

        out_dict['mask'] = mask

        if return_lines:
            for i in range(repeats):
                out_dims[0] = _ns[i]
                streaks = ArrayWrapper.from_ptr(<void *>_outs[i]).to_ndarray(2, out_dims, np.NPY_FLOAT64)
                line_dict[i] = np.PyArray_Compress(streaks, streaks[:, 0], 0, <np.ndarray>NULL)

            out_dict['lines'] = line_dict
        else:
            for i in range(repeats):
                free(_outs[i])

        free(_ns); free(_outs)

        return out_dict

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

def draw_lines(image: np.ndarray, lines: np.ndarray, max_val: cython.uint=255, dilation: cython.uint=0) -> np.ndarray:
    """Draw thick lines with variable thickness. The lines must follow
    the LSD convention, see the parameters for more info.

    Parameters
    ----------
    image : np.ndarray
        Image array.
    lines : np.ndarray
        An array of the detected lines. Must have a shape of (`N`, 7),
        where `N` is the number of lines. Each line is comprised of
        7 parameters as follows:

        * `[x1, y1]`, `[x2, y2]` : The coordinates of the line's
          ends.
        * `width` : Line's width.
        * `p` : Angle precision [0, 1] given by angle tolerance
          over 180 degree.
        * `-log10(NFA)` : Number of false alarms.
    max_val : int, optional
        Maximum value of the line mask.
    dilation : int, optional
        Size of the binary dilation applied to the output image.

    Returns
    -------
    image : np.ndarray
        Output image with the lines drawn.

    See Also
    --------
    LSD : Line Segment Detector.
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
    cdef int fail

    with nogil:
        fail = draw_lines_c(_image, _Y, _X, max_val, _lines, _n_lines, dilation)
    if fail:
        raise RuntimeError('C backend exited with error.')    
    return image

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

def subtract_background(data: np.ndarray, mask: np.ndarray, whitefield: np.ndarray, good_frames: np.ndarray, 
                        num_threads: cython.uint=1, out: np.ndarray=None) -> np.ndarray:
    data = check_array(data, np.NPY_UINT32)
    mask = check_array(mask, np.NPY_BOOL)
    whitefield = check_array(whitefield, np.NPY_FLOAT32)
    good_frames = check_array(good_frames, np.NPY_UINT32)

    cdef int i, ii, j, k
    cdef float res, w0, w1

    if out is None:
        out = <np.ndarray>np.PyArray_SimpleNew(data.ndim, data.shape, np.NPY_FLOAT32)
    else:
        out = check_array(out, np.NPY_FLOAT32)

    cdef np.uint32_t[:, :, ::1] _data = data
    cdef np.npy_bool[:, :, ::1] _mask = mask
    cdef np.float32_t[:, ::1] _whitefield = whitefield
    cdef np.uint32_t[::1] _good_frames = good_frames
    cdef np.float32_t[:, :, ::1] _out = out

    cdef int n_frames = good_frames.size
    num_threads = n_frames if <int>num_threads > n_frames else <int>num_threads
    for i in prange(n_frames, schedule='guided', num_threads=num_threads, nogil=True):
        ii = _good_frames[i]
        w0 = 0.0; w1 = 0.0
        for j in range(_data.shape[1]):
            for k in range(_data.shape[2]):
                if _mask[ii, j, k]:
                    w0 = w0 + <float>_data[ii, j, k] * _whitefield[j, k]
                    w1 = w1 + _whitefield[j, k] * _whitefield[j, k]
        w0 = w0 / w1 if w1 > 0.0 else 1.0
        for j in range(_data.shape[1]):
            for k in range(_data.shape[2]):
                if _mask[ii, j, k]:
                    res = <float>_data[ii, j, k] - w0 * _whitefield[j, k]
                    _out[ii, j, k] = res if res > 0.0 else 0.0
                else:
                    _out[ii, j, k] = 0.0

    return out