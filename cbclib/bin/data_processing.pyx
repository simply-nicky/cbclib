cimport numpy as np
import numpy as np
import cython
from libc.math cimport log
from libc.math cimport sqrt, pi
from libc.string cimport memcmp
from libc.stdlib cimport malloc, free

# Set the cleanup routine
cdef void _cleanup():
    fftw_cleanup()
    fftw_cleanup_threads()

fftw_init_threads()

Py_AtExit(_cleanup)

# Numpy must be initialized. When using numpy from C or Cython you must
# *ALWAYS* do that, or you will have segfaults
np.import_array()

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

def next_fast_len(target: cython.uint, backend: str='numpy') -> cython.uint:
    r"""Find the next fast size of input data to fft, for zero-padding, etc.
    FFT algorithms gain their speed by a recursive divide and conquer strategy.
    This relies on efficient functions for small prime factors of the input length.
    Thus, the transforms are fastest when using composites of the prime factors handled
    by the fft implementation. If there are efficient functions for all radices <= n,
    then the result will be a number x >= target with only prime factors < n. (Also
    known as n-smooth numbers)

    Parameters
    ----------
    target : int
        Length to start searching from. Must be a positive integer.
    backend : {'fftw', 'numpy'}, optional
        Find n-smooth number for the FFT implementation from the specified
        library.

    Returns
    -------
    n : int
        The smallest fast length greater than or equal to `target`.
    """
    if target < 0:
        raise ValueError('Target length must be positive')
    if backend == 'fftw':
        return next_fast_len_fftw(target)
    elif backend == 'numpy':
        return good_size(target)
    else:
        raise ValueError('{:s} is invalid backend'.format(backend))

def fft_convolve(array: np.ndarray, kernel: np.ndarray, axis: cython.int=-1,
                 mode: str='constant', cval: cython.double=0.0, backend: str='numpy',
                 num_threads: cython.uint=1) -> np.ndarray:
    """Convolve a multi-dimensional `array` with one-dimensional `kernel` along the
    `axis` by means of FFT. Output has the same size as `array`.

    Parameters
    ----------
    array : numpy.ndarray
        Input array.
    kernel : numpy.ndarray
        Kernel array.
    axis : int, optional
        Array axis along which convolution is performed.
    mode : {'constant', 'nearest', 'mirror', 'reflect', 'wrap'}, optional
        The mode parameter determines how the input array is extended when the filter
        overlaps a border. Default value is 'constant'. The valid values and their behavior
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
    backend : {'fftw', 'numpy'}, optional
        Choose backend library for the FFT implementation.
    num_threads : int, optional
        Number of threads.

    Returns
    -------
    out : numpy.ndarray
        A multi-dimensional array containing the discrete linear
        convolution of `array` with `kernel`.
    """
    array = check_array(array, np.NPY_FLOAT64)
    kernel = check_array(kernel, np.NPY_FLOAT64)

    cdef int fail = 0
    cdef int ndim = array.ndim
    axis = axis if axis >= 0 else ndim + axis
    axis = axis if axis <= ndim - 1 else ndim - 1
    cdef np.npy_intp ksize = np.PyArray_DIM(kernel, 0)
    cdef int _mode = extend_mode_to_code(mode)
    cdef np.npy_intp *dims = array.shape
    cdef unsigned long *_dims = <unsigned long *>dims

    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(ndim, dims, np.NPY_FLOAT64)
    cdef double *_out = <double *>np.PyArray_DATA(out)
    cdef double *_inp = <double *>np.PyArray_DATA(array)
    cdef double *_krn = <double *>np.PyArray_DATA(kernel)
    with nogil:
        if backend == 'fftw':
            fail = fft_convolve_fftw(_out, _inp, ndim, _dims, _krn, ksize, axis, _mode, cval, num_threads)
        elif backend == 'numpy':
            fail = fft_convolve_np(_out, _inp, ndim, _dims, _krn, ksize, axis, _mode, cval, num_threads)
        else:
            raise ValueError('{:s} is invalid backend'.format(backend))
    if fail:
        raise RuntimeError('C backend exited with error.')
    return out

def gaussian_filter(inp: np.ndarray, sigma: object, order: object=0, mode: str='reflect',
                    cval: cython.double=0., truncate: cython.double=4., backend: str='numpy',
                    num_threads: cython.uint=1) -> np.ndarray:
    r"""Multidimensional Gaussian filter. The multidimensional filter is implemented as
    a sequence of 1-D FFT convolutions.

    Parameters
    ----------
    inp : np.ndarray
        The input array.
    sigma : float or list of floats
        Standard deviation for Gaussian kernel. The standard deviations of the Gaussian
        filter are given for each axis as a sequence, or as a single number, in which case
        it is equal for all axes.
    order : int or list of ints, optional
        The order of the filter along each axis is given as a sequence of integers, or as
        a single number. An order of 0 corresponds to convolution with a Gaussian kernel.
        A positive order corresponds to convolution with that derivative of a Gaussian.
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
    truncate : float, optional
        Truncate the filter at this many standard deviations. Default is 4.0.
    backend : {'fftw', 'numpy'}, optional
        Choose backend library for the FFT implementation.
    num_threads : int, optional
        Number of threads.
    
    Returns
    -------
    out : np.ndarray
        Returned array of same shape as `input`.
    """
    inp = check_array(inp, np.NPY_FLOAT64)

    cdef int ndim = inp.ndim
    cdef np.ndarray sigmas = normalize_sequence(sigma, ndim, np.NPY_FLOAT64)
    cdef np.ndarray orders = normalize_sequence(order, ndim, np.NPY_UINT32)

    cdef int fail = 0
    cdef np.npy_intp *dims = inp.shape
    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(ndim, dims, np.NPY_FLOAT64)
    cdef double *_out = <double *>np.PyArray_DATA(out)
    cdef double *_inp = <double *>np.PyArray_DATA(inp)
    cdef unsigned long *_dims = <unsigned long *>dims
    cdef double *_sig = <double *>np.PyArray_DATA(sigmas)
    cdef unsigned *_ord = <unsigned *>np.PyArray_DATA(orders)
    cdef int _mode = extend_mode_to_code(mode)
    with nogil:
        if backend == 'fftw':
            fail = gauss_filter(_out, _inp, ndim, _dims, _sig, _ord, _mode, cval, truncate, num_threads, fft_convolve_fftw)
        elif backend == 'numpy':
            fail = gauss_filter(_out, _inp, ndim, _dims, _sig, _ord, _mode, cval, truncate, num_threads, fft_convolve_np)
        else:
            raise ValueError('{:s} is invalid backend'.format(backend))
    if fail:
        raise RuntimeError('C backend exited with error.')
    return out

def gaussian_kernel(sigma: double, order: cython.uint=0, truncate: cython.double=4.) -> np.ndarray:
    """Discrete Gaussian kernel.
    
    Parameters
    ----------
    sigma : float
        Standard deviation for Gaussian kernel.
    order : int, optional
        The order of the filter. An order of 0 corresponds to convolution with a
        Gaussian kernel. A positive order corresponds to convolution with that
        derivative of a Gaussian. Default is 0.
    truncate : float, optional
        Truncate the filter at this many standard deviations. Default is 4.0.
    
    Returns
    -------
    krn : np.ndarray
        Gaussian kernel.
    """
    cdef np.npy_intp radius = <np.npy_intp>(sigma * truncate)
    cdef np.npy_intp *dims = [2 * radius + 1,]
    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(1, dims, np.NPY_FLOAT64)
    cdef double *_out = <double *>np.PyArray_DATA(out)
    with nogil:
        gauss_kernel1d(_out, sigma, order, dims[0])
    return out

def gaussian_gradient_magnitude(inp: np.ndarray, sigma: object, mode: str='reflect',
                                cval: cython.double=0., truncate: cython.double=4.,
                                backend: str='numpy', num_threads: cython.uint=1) -> np.ndarray:
    r"""Multidimensional gradient magnitude using Gaussian derivatives. The multidimensional
    filter is implemented as a sequence of 1-D FFT convolutions.

    Parameters
    ----------
    inp : np.ndarray
        The input array.
    sigma : float or list of floats
        The standard deviations of the Gaussian filter are given for each axis as a sequence,
        or as a single number, in which case it is equal for all axes.
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
    truncate : float, optional
        Truncate the filter at this many standard deviations. Default is 4.0.
    backend : {'fftw', 'numpy'}, optional
        Choose backend library for the FFT implementation.
    num_threads : int, optional
        Number of threads.
    """
    inp = check_array(inp, np.NPY_FLOAT64)

    cdef int ndim = inp.ndim
    cdef np.ndarray sigmas = normalize_sequence(sigma, ndim, np.NPY_FLOAT64)
    
    cdef int fail = 0
    cdef np.npy_intp *dims = inp.shape
    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(ndim, dims, np.NPY_FLOAT64)
    cdef double *_out = <double *>np.PyArray_DATA(out)
    cdef double *_inp = <double *>np.PyArray_DATA(inp)
    cdef unsigned long *_dims = <unsigned long *>dims
    cdef double *_sig = <double *>np.PyArray_DATA(sigmas)
    cdef int _mode = extend_mode_to_code(mode)
    with nogil:
        if backend == 'fftw':
            fail = gauss_grad_mag(_out, _inp, ndim, _dims, _sig, _mode, cval, truncate, num_threads, fft_convolve_fftw)
        elif backend == 'numpy':
            fail = gauss_grad_mag(_out, _inp, ndim, _dims, _sig, _mode, cval, truncate, num_threads, fft_convolve_np)
        else:
            raise ValueError('{:s} is invalid backend'.format(backend))
    if fail:
        raise RuntimeError('C backend exited with error.')
    return out

def median(data: np.ndarray, mask: np.ndarray, axis: cython.int=0,
           num_threads: cython.uint=1) -> np.ndarray:
    """Calculate a median along the `axis`.

    Parameters
    ----------
    data : numpy.ndarray
        Intensity frames.
    mask : numpy.ndarray
        Bad pixel mask.
    axis : int, optional
        Array axis along which median values are calculated.
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
    axis = axis if axis >= 0 else ndim + axis
    axis = axis if axis <= ndim - 1 else ndim - 1

    if mask is None:
        mask = <np.ndarray>np.PyArray_SimpleNew(ndim, data.shape, np.NPY_BOOL)
        np.PyArray_FILLWBYTE(mask, 1)
    else:
        mask = check_array(mask, np.NPY_BOOL)
        if memcmp(data.shape, mask.shape, ndim * sizeof(np.npy_intp)):
            raise ValueError('mask and data arrays must have identical shapes')

    cdef unsigned long *_dims = <unsigned long *>data.shape

    cdef np.npy_intp *odims = <np.npy_intp *>malloc((ndim - 1) * sizeof(np.npy_intp))
    if odims is NULL:
        raise MemoryError('not enough memory')
    cdef int i
    for i in range(axis):
        odims[i] = data.shape[i]
    for i in range(axis + 1, ndim):
        odims[i - 1] = data.shape[i]

    cdef int type_num = np.PyArray_TYPE(data)
    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(ndim - 1, odims, type_num)
    cdef void *_out = <void *>np.PyArray_DATA(out)
    cdef void *_data = <void *>np.PyArray_DATA(data)
    cdef unsigned char *_mask = <unsigned char *>np.PyArray_DATA(mask)

    with nogil:
        if type_num == np.NPY_FLOAT64:
            fail = median_c(_out, _data, _mask, ndim, _dims, 8, axis, compare_double, num_threads)
        elif type_num == np.NPY_FLOAT32:
            fail = median_c(_out, _data, _mask, ndim, _dims, 4, axis, compare_float, num_threads)
        elif type_num == np.NPY_INT32:
            fail = median_c(_out, _data, _mask, ndim, _dims, 4, axis, compare_int, num_threads)
        elif type_num == np.NPY_UINT32:
            fail = median_c(_out, _data, _mask, ndim, _dims, 4, axis, compare_uint, num_threads)
        else:
            raise TypeError('data argument has incompatible type: {:s}'.format(data.dtype))
    if fail:
        raise RuntimeError('C backend exited with error.')

    free(odims)
    return out

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