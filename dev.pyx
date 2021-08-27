#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
cimport numpy as np
import numpy as np
import cython
from libc.stdlib cimport free, malloc, calloc
from libc.string cimport memcmp
from cpython.ref cimport Py_INCREF
from cython.parallel import prange

# Numpy must be initialized. When using numpy from C or Cython you must
# *ALWAYS* do that, or you will have segfaults
np.import_array()

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

    def __cinit__(self, double scale=0.8, double sigma_scale=0.6, double log_eps=0.,
                  double ang_th=45.0, double density_th=0.7, double quant=2.0):
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

    def __init__(self, scale=0.8, sigma_scale=0.6, log_eps=0, ang_th=45.0, density_th=0.7, quant=2.0):
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

    cpdef dict detect(self, np.ndarray image):
        """Perform the streak detection on `image`.

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
        if image.ndim != 2:
            raise ValueError('Image must be a 2D array.')
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
            fail = LineSegmentDetection(&_out, &_n_out, _img, _img_x, _img_y,
                                        self.scale, self.sigma_scale, self.quant,
                                        self.ang_th, self.log_eps, self.density_th, N_BINS,
                                        &_reg_img, &_reg_x, &_reg_y)

        if fail:
            raise RuntimeError("LSD execution finished with an error.")
        
        cdef np.npy_intp *out_dims = [_n_out, LINE_SIZE]
        cdef np.ndarray out = ArrayWrapper.from_ptr(<void *>_out).to_ndarray(2, out_dims, np.NPY_FLOAT64)

        cdef np.npy_intp *reg_dims = [_reg_y, _reg_x,]
        cdef np.ndarray reg_img = ArrayWrapper.from_ptr(<void *>_reg_img).to_ndarray(2, reg_dims, np.NPY_INT32)

        return {'lines': out, 'labels': reg_img}

    cpdef np.ndarray mask(self, np.ndarray image, unsigned int max_val=1, unsigned int dilation=0, unsigned int num_threads=1):
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
        cdef double *img_ptr = <double *>np.PyArray_DATA(image)
        cdef int X = <int>image.shape[ndim - 1]
        cdef int Y = <int>image.shape[ndim - 2]

        cdef np.ndarray mask = np.PyArray_ZEROS(ndim, image.shape, np.NPY_UINT32, 0)
        cdef unsigned int *msk_ptr = <unsigned int *>np.PyArray_DATA(mask)
        cdef double *out
        cdef int n_out, fail, i

        cdef unsigned int repeats = image.size / X / Y
        num_threads = repeats if num_threads > repeats else num_threads
        for i in prange(repeats, schedule='guided', num_threads=num_threads, nogil=True):
            fail = LineSegmentDetection(&out, &n_out, img_ptr + i * X * Y, X, Y,
                                        self.scale, self.sigma_scale, self.quant,
                                        self.ang_th, self.log_eps, self.density_th, N_BINS,
                                        NULL, NULL, NULL)
            draw_lines_c(msk_ptr + i * X * Y, X, Y, max_val, out, n_out, dilation)
        return mask

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

def median_filter(data: np.ndarray, size: object=None, footprint: np.ndarray=None, mask: np.ndarray=None,
                  mode: str='reflect', cval: cython.double=0., num_threads: cython.uint=1) -> np.ndarray:
    """Calculate a multidimensional median filter.

    Parameters
    ----------
    data : numpy.ndarray
        Intensity frames.
    size : scalar or tuple, optional
        See footprint, below. Ignored if footprint is given.
    footprint : numpy.ndarray, optional
        Either size or footprint must be defined. size gives the shape that is taken from the
        input array, at every element position, to define the input to the filter function.
        footprint is a boolean array that specifies (implicitly) a shape, but also which of
        the elements within this shape will get passed to the filter function. Thus size=(n,m)
        is equivalent to footprint=np.ones((n,m)). We adjust size to the number of dimensions
        of the input array, so that, if the input array is shape (10,10,10), and size is 2,
        then the actual size used is (2,2,2). When footprint is given, size is ignored.
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
    if not np.PyArray_IS_C_CONTIGUOUS(data):
        data = np.PyArray_GETCONTIGUOUS(data)

    cdef int ndim = data.ndim
    cdef np.npy_intp *dims = data.shape

    if mask is None:
        mask = <np.ndarray>np.PyArray_SimpleNew(ndim, dims, np.NPY_BOOL)
        np.PyArray_FILLWBYTE(mask, 1)
    else:
        mask = check_array(mask, np.NPY_BOOL)

    if size is None and footprint is None:
        raise ValueError('size or footprint must be provided.')

    cdef unsigned long *_fsize
    cdef np.ndarray fsize
    if size is None:
        _fsize = <unsigned long *>footprint.shape
    else:
        fsize = normalize_sequence(size, ndim, np.NPY_INTP)
        _fsize = <unsigned long *>np.PyArray_DATA(fsize)

    if footprint is None:
        footprint = <np.ndarray>np.PyArray_SimpleNew(ndim, <np.npy_intp *>_fsize, np.NPY_BOOL)
        np.PyArray_FILLWBYTE(footprint, 1)
    cdef unsigned char *_fmask = <unsigned char *>np.PyArray_DATA(footprint)

    cdef unsigned long *_dims = <unsigned long *>dims
    cdef int type_num = np.PyArray_TYPE(data)
    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(ndim, dims, type_num)
    cdef void *_out = <void *>np.PyArray_DATA(out)
    cdef void *_data = <void *>np.PyArray_DATA(data)
    cdef unsigned char *_mask = <unsigned char *>np.PyArray_DATA(mask)
    cdef int _mode = extend_mode_to_code(mode)
    cdef void *_cval = <void *>&cval

    with nogil:
        if type_num == np.NPY_FLOAT64:
            fail = median_filter_c(_out, _data, _mask, ndim, _dims, 8, _fsize, _fmask, _mode, _cval, compare_double, num_threads)
        elif type_num == np.NPY_FLOAT32:
            fail = median_filter_c(_out, _data, _mask, ndim, _dims, 4, _fsize, _fmask, _mode, _cval, compare_float, num_threads)
        elif type_num == np.NPY_INT32:
            fail = median_filter_c(_out, _data, _mask, ndim, _dims, 4, _fsize, _fmask, _mode, _cval, compare_int, num_threads)
        elif type_num == np.NPY_UINT32:
            fail = median_filter_c(_out, _data, _mask, ndim, _dims, 4, _fsize, _fmask, _mode, _cval, compare_uint, num_threads)
        else:
            raise TypeError('data argument has incompatible type: {:s}'.format(str(data.dtype)))

    if fail:
        raise RuntimeError('C backend exited with error.')
    return out

def maximum_filter(data: np.ndarray, size: object=None, footprint: np.ndarray=None, mask: np.ndarray=None,
                   mode: str='reflect', cval: cython.double=0., num_threads: cython.uint=1) -> np.ndarray:
    """Calculate a multidimensional maximum filter.

    Parameters
    ----------
    data : numpy.ndarray
        Intensity frames.
    size : scalar or tuple, optional
        See footprint, below. Ignored if footprint is given.
    footprint : numpy.ndarray, optional
        Either size or footprint must be defined. size gives the shape that is taken from the
        input array, at every element position, to define the input to the filter function.
        footprint is a boolean array that specifies (implicitly) a shape, but also which of
        the elements within this shape will get passed to the filter function. Thus size=(n,m)
        is equivalent to footprint=np.ones((n,m)). We adjust size to the number of dimensions
        of the input array, so that, if the input array is shape (10,10,10), and size is 2,
        then the actual size used is (2,2,2). When footprint is given, size is ignored.
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
    if not np.PyArray_IS_C_CONTIGUOUS(data):
        data = np.PyArray_GETCONTIGUOUS(data)

    cdef int ndim = data.ndim
    cdef np.npy_intp *dims = data.shape

    if mask is None:
        mask = <np.ndarray>np.PyArray_SimpleNew(ndim, dims, np.NPY_BOOL)
        np.PyArray_FILLWBYTE(mask, 1)
    else:
        mask = check_array(mask, np.NPY_BOOL)

    if size is None and footprint is None:
        raise ValueError('size or footprint must be provided.')

    cdef unsigned long *_fsize
    cdef np.ndarray fsize
    if size is None:
        _fsize = <unsigned long *>footprint.shape
    else:
        fsize = normalize_sequence(size, ndim, np.NPY_INTP)
        _fsize = <unsigned long *>np.PyArray_DATA(fsize)

    if footprint is None:
        footprint = <np.ndarray>np.PyArray_SimpleNew(ndim, <np.npy_intp *>_fsize, np.NPY_BOOL)
        np.PyArray_FILLWBYTE(footprint, 1)
    cdef unsigned char *_fmask = <unsigned char *>np.PyArray_DATA(footprint)

    cdef unsigned long *_dims = <unsigned long *>dims
    cdef int type_num = np.PyArray_TYPE(data)
    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(ndim, dims, type_num)
    cdef void *_out = <void *>np.PyArray_DATA(out)
    cdef void *_data = <void *>np.PyArray_DATA(data)
    cdef unsigned char *_mask = <unsigned char *>np.PyArray_DATA(mask)
    cdef int _mode = extend_mode_to_code(mode)
    cdef void *_cval = <void *>&cval

    with nogil:
        if type_num == np.NPY_FLOAT64:
            fail = maximum_filter_c(_out, _data, _mask, ndim, _dims, 8, _fsize, _fmask, _mode, _cval, compare_double, num_threads)
        elif type_num == np.NPY_FLOAT32:
            fail = maximum_filter_c(_out, _data, _mask, ndim, _dims, 4, _fsize, _fmask, _mode, _cval, compare_float, num_threads)
        elif type_num == np.NPY_INT32:
            fail = maximum_filter_c(_out, _data, _mask, ndim, _dims, 4, _fsize, _fmask, _mode, _cval, compare_int, num_threads)
        elif type_num == np.NPY_UINT32:
            fail = maximum_filter_c(_out, _data, _mask, ndim, _dims, 4, _fsize, _fmask, _mode, _cval, compare_uint, num_threads)
        else:
            raise TypeError('data argument has incompatible type: {:s}'.format(str(data.dtype)))

    if fail:
        raise RuntimeError('C backend exited with error.')
    return out

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

    with nogil:
        fail = draw_lines_c(_image, _Y, _X, max_val, _lines, _n_lines, dilation)
    if fail:
        raise RuntimeError('C backend exited with error.')    
    return image
