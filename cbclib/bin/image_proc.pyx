import numpy as np
import cython
from libc.math cimport log, sqrt, pi
from libc.string cimport memcmp
from libc.stdlib cimport malloc, free
from cython.parallel import prange
from .line_detector cimport ArrayWrapper

# Set the cleanup routine
cdef void _cleanup():
    fftw_cleanup()
    fftw_cleanup_threads()

fftw_init_threads()

Py_AtExit(_cleanup)

# Numpy must be initialized. When using numpy from C or Cython you must
# *ALWAYS* do that, or you will have segfaults
np.import_array()

def next_fast_len(unsigned target, str backend='numpy'):
    r"""Find the next fast size of input data to fft, for zero-padding, etc.
    FFT algorithms gain their speed by a recursive divide and conquer strategy.
    This relies on efficient functions for small prime factors of the input length.
    Thus, the transforms are fastest when using composites of the prime factors handled
    by the fft implementation. If there are efficient functions for all radices <= n,
    then the result will be a number x >= target with only prime factors < n. (Also
    known as n-smooth numbers)

    Args:
        target (int) : Length to start searching from. Must be a positive integer.
        backend (str) : Find n-smooth number for the FFT implementation from the numpy
            ('numpy') or FFTW ('fftw') library.

    Returns:
        int : The smallest fast length greater than or equal to `target`.
    """
    if target < 0:
        raise ValueError('Target length must be positive')
    if backend == 'fftw':
        return next_fast_len_fftw(target)
    elif backend == 'numpy':
        return good_size(target)
    else:
        raise ValueError('{:s} is invalid backend'.format(backend))

def fft_convolve(np.ndarray array not None, np.ndarray kernel not None, int axis=-1,
                 str mode='constant', double cval=0.0, str backend='numpy',
                 unsigned num_threads=1):
    """Convolve a multi-dimensional `array` with one-dimensional `kernel` along the
    `axis` by means of FFT. Output has the same size as `array`.

    Args:
        array (numpy.ndarray) : Input array.
        kernel (numpy.ndarray) : Kernel array.
        axis (int) : Array axis along which convolution is performed.
        mode (str) : The mode parameter determines how the input array is extended
            when the filter overlaps a border. Default value is 'constant'. The
            valid values and their behavior is as follows:

            * 'constant', (k k k k | a b c d | k k k k) : The input is extended by
              filling all values beyond the edge with the same constant value, defined
              by the `cval` parameter.
            * 'nearest', (a a a a | a b c d | d d d d) : The input is extended by
              replicating the last pixel.
            * 'mirror', (c d c b | a b c d | c b a b) : The input is extended by
              reflecting about the center of the last pixel. This mode is also sometimes
              referred to as whole-sample symmetric.
            * 'reflect', (d c b a | a b c d | d c b a) : The input is extended by
              reflecting about the edge of the last pixel. This mode is also sometimes
              referred to as half-sample symmetric.
            * 'wrap', (a b c d | a b c d | a b c d) : The input is extended by wrapping
              around to the opposite edge.

        cval (float) :  Value to fill past edges of input if mode is 'constant'. Default
            is 0.0.
        backend (str) : Choose between numpy ('numpy') or FFTW ('fftw') library for the FFT
            implementation.
        num_threads (int) : Number of threads used in the calculations.

    Returns:
        numpy.ndarray : A multi-dimensional array containing the discrete linear
        convolution of `array` with `kernel`.
    """
    cdef int fail = 0
    cdef int ndim = array.ndim
    axis = axis if axis >= 0 else ndim + axis
    axis = axis if axis <= ndim - 1 else ndim - 1
    cdef np.npy_intp ksize = np.PyArray_DIM(kernel, 0)
    cdef int _mode = extend_mode_to_code(mode)
    cdef np.npy_intp *dims = array.shape
    cdef unsigned long *_dims = <unsigned long *>dims

    cdef int type_num
    if np.PyArray_ISCOMPLEX(array) or np.PyArray_ISCOMPLEX(kernel):
        type_num = np.NPY_COMPLEX128
    else:
        type_num = np.NPY_FLOAT64

    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(ndim, dims, type_num)
    cdef void *_out = np.PyArray_DATA(out)
    cdef void *_inp
    cdef void *_krn

    if np.PyArray_ISCOMPLEX(array) or np.PyArray_ISCOMPLEX(kernel):
        array = check_array(array, np.NPY_COMPLEX128)
        kernel = check_array(kernel, np.NPY_COMPLEX128)

        _inp = np.PyArray_DATA(array)
        _krn = np.PyArray_DATA(kernel)

        with nogil:
            if backend == 'fftw':
                fail = cfft_convolve_fftw(<double complex *>_out, <double complex *>_inp, ndim,
                                          _dims, <double complex *>_krn, ksize, axis, _mode,
                                          <double complex>cval, num_threads)
            elif backend == 'numpy':
                fail = cfft_convolve_np(<double complex *>_out, <double complex *>_inp, ndim,
                                        _dims, <double complex *>_krn, ksize, axis, _mode,
                                        <double complex>cval, num_threads)
            else:
                raise ValueError('{:s} is invalid backend'.format(backend))
    else:
        array = check_array(array, np.NPY_FLOAT64)
        kernel = check_array(kernel, np.NPY_FLOAT64)

        _inp = np.PyArray_DATA(array)
        _krn = np.PyArray_DATA(kernel)

        with nogil:
            if backend == 'fftw':
                fail = rfft_convolve_fftw(<double *>_out, <double *>_inp, ndim, _dims,
                                          <double *>_krn, ksize, axis, _mode, cval, num_threads)
            elif backend == 'numpy':
                fail = rfft_convolve_np(<double *>_out, <double *>_inp, ndim, _dims,
                                        <double *>_krn, ksize, axis, _mode, cval, num_threads)
            else:
                raise ValueError('{:s} is invalid backend'.format(backend))

    if fail:
        raise RuntimeError('C backend exited with error.')
    return out

def gaussian_kernel(double sigma, unsigned order=0, double truncate=4.):
    """Discrete Gaussian kernel.
    
    Args:
        sigma (float) : Standard deviation for Gaussian kernel.
        order (int) : The order of the filter. An order of 0 corresponds to
            convolution with a Gaussian kernel. A positive order corresponds
            to convolution with that derivative of a Gaussian. Default is 0.
        truncate (float) : Truncate the filter at this many standard deviations.
            Default is 4.0.
    
    Returns:
        numpy.ndarray : Gaussian kernel.
    """
    cdef np.npy_intp radius = <np.npy_intp>(sigma * truncate)
    cdef np.npy_intp *dims = [2 * radius + 1,]
    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(1, dims, np.NPY_FLOAT64)
    cdef double *_out = <double *>np.PyArray_DATA(out)
    with nogil:
        gauss_kernel1d(_out, sigma, order, dims[0], 1)
    return out

def gaussian_filter(np.ndarray inp not None, object sigma not None, object order not None=0,
                    str mode='reflect', double cval=0., double truncate=4., str backend='numpy',
                    unsigned num_threads=1):
    r"""Multidimensional Gaussian filter. The multidimensional filter is implemented as
    a sequence of 1-D FFT convolutions.

    Args:
        inp (numpy.ndarray) : The input array.
        sigma (Union[float, List[float]]): Standard deviation for Gaussian kernel. The standard
            deviations of the Gaussian filter are given for each axis as a sequence, or as a
            single number, in which case it is equal for all axes.
        order (Union[int, List[int]]): The order of the filter along each axis is given as a
            sequence of integers, or as a single number. An order of 0 corresponds to convolution
            with a Gaussian kernel. A positive order corresponds to convolution with that
            derivative of a Gaussian.
        mode (str) : The mode parameter determines how the input array is extended when the
            filter overlaps a border. Default value is 'reflect'. The valid values and their
            behavior is as follows:

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

        cval (float) : Value to fill past edges of input if mode is 'constant'. Default is 0.0.
        truncate (float) : Truncate the filter at this many standard deviations. Default is 4.0.
        backend (str) : Choose between numpy ('numpy') or FFTW ('fftw') backend library for the
            FFT implementation.
        num_threads (int) : Number of threads.
    
    Returns:
        numpy.ndarray : Returned array of the same shape as `input`.
    """

    cdef int ndim = inp.ndim
    cdef np.ndarray sigmas = normalize_sequence(sigma, ndim, np.NPY_FLOAT64)
    cdef np.ndarray orders = normalize_sequence(order, ndim, np.NPY_UINT32)
    cdef double *_sig = <double *>np.PyArray_DATA(sigmas)
    cdef unsigned *_ord = <unsigned *>np.PyArray_DATA(orders)

    cdef int n
    for n in range(ndim):
        if inp.shape[n] == 1:
            sigmas[n] = 0.0

    cdef int fail = 0
    cdef int _mode = extend_mode_to_code(mode)
    cdef np.npy_intp *dims = inp.shape
    cdef unsigned long *_dims = <unsigned long *>dims

    cdef int type_num
    if np.PyArray_ISCOMPLEX(inp):
        type_num = np.NPY_COMPLEX128
    else:
        type_num = np.NPY_FLOAT64

    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(ndim, dims, type_num)
    cdef void *_out = np.PyArray_DATA(out)

    cdef void *_inp
    if np.PyArray_ISCOMPLEX(inp):
        inp = check_array(inp, np.NPY_COMPLEX128)
        _inp = <double *>np.PyArray_DATA(inp)

        with nogil:
            if backend == 'fftw':
                fail = gauss_filter_c(<double complex *>_out, <double complex *>_inp,
                                      ndim, _dims, _sig, _ord, _mode, <double complex>cval,
                                      truncate, num_threads, cfft_convolve_fftw)
            elif backend == 'numpy':
                fail = gauss_filter_c(<double complex *>_out, <double complex *>_inp,
                                      ndim, _dims, _sig, _ord, _mode, <double complex>cval,
                                      truncate, num_threads, cfft_convolve_np)
            else:
                raise ValueError('{:s} is invalid backend'.format(backend))

    else:
        inp = check_array(inp, np.NPY_FLOAT64)
        _inp = <double *>np.PyArray_DATA(inp)

        with nogil:
            if backend == 'fftw':
                fail = gauss_filter_r(<double *>_out, <double *>_inp, ndim, _dims, _sig,
                                      _ord, _mode, cval, truncate, num_threads, rfft_convolve_fftw)
            elif backend == 'numpy':
                fail = gauss_filter_r(<double *>_out, <double *>_inp, ndim, _dims, _sig,
                                      _ord, _mode, cval, truncate, num_threads, rfft_convolve_np)
            else:
                raise ValueError('{:s} is invalid backend'.format(backend))

    if fail:
        raise RuntimeError('C backend exited with error.')
    return out

def gaussian_gradient_magnitude(np.ndarray inp not None, object sigma not None, str mode='reflect',
                                double cval=0.0, double truncate=4.0, str backend='numpy',
                                unsigned num_threads=1):
    r"""Multidimensional gradient magnitude using Gaussian derivatives. The multidimensional
    filter is implemented as a sequence of 1-D FFT convolutions.

    Args:
        inp (numpy.ndarray) : The input array.
        sigma (Union[float, List[float]]): Standard deviation for Gaussian kernel. The standard
            deviations of the Gaussian filter are given for each axis as a sequence, or as a
            single number, in which case it is equal for all axes.
        mode (str) : The mode parameter determines how the input array is extended when the
            filter overlaps a border. Default value is 'reflect'. The valid values and their
            behavior is as follows:

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

        cval (float) : Value to fill past edges of input if mode is ‘constant’. Default is 0.0.
        truncate (float) : Truncate the filter at this many standard deviations. Default is 4.0.
        backend (str) : Choose between numpy ('numpy') or FFTW ('fftw') backend library for the
            FFT implementation.
        num_threads (int) : Number of threads.

    Returns:
        numpy.ndarray : Gaussian gradient magnitude array. The array is the same shape as `input`.
    """
    cdef int ndim = inp.ndim
    cdef np.ndarray sigmas = normalize_sequence(sigma, ndim, np.NPY_FLOAT64)
    cdef double *_sig = <double *>np.PyArray_DATA(sigmas)

    cdef int n
    for n in range(ndim):
        if inp.shape[n] == 1:
            sigmas[n] = 0.0

    cdef int fail = 0
    cdef int _mode = extend_mode_to_code(mode)
    cdef np.npy_intp *dims = inp.shape
    cdef unsigned long *_dims = <unsigned long *>dims

    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(ndim, dims, np.NPY_FLOAT64)
    cdef double *_out = <double *>np.PyArray_DATA(out)

    cdef void *_inp
    if np.PyArray_ISCOMPLEX(inp):
        inp = check_array(inp, np.NPY_COMPLEX128)
        _inp = <double *>np.PyArray_DATA(inp)

        with nogil:
            if backend == 'fftw':
                fail = gauss_grad_mag_c(_out, <double complex *>_inp, ndim, _dims, _sig,
                                        _mode, <double complex>cval, truncate, num_threads,
                                        cfft_convolve_fftw)
            elif backend == 'numpy':
                fail = gauss_grad_mag_c(_out, <double complex *>_inp, ndim, _dims, _sig,
                                        _mode, <double complex>cval, truncate, num_threads,
                                        cfft_convolve_np)
            else:
                raise ValueError('{:s} is invalid backend'.format(backend))

    else:
        inp = check_array(inp, np.NPY_FLOAT64)
        _inp = <double *>np.PyArray_DATA(inp)

        with nogil:
            if backend == 'fftw':
                fail = gauss_grad_mag_r(_out, <double *>_inp, ndim, _dims, _sig, _mode,
                                        cval, truncate, num_threads, rfft_convolve_fftw)
            elif backend == 'numpy':
                fail = gauss_grad_mag_r(_out, <double *>_inp, ndim, _dims, _sig, _mode,
                                        cval, truncate, num_threads, rfft_convolve_np)
            else:
                raise ValueError('{:s} is invalid backend'.format(backend))
    
    if fail:
        raise RuntimeError('C backend exited with error.')
    return out

def median(np.ndarray data not None, np.ndarray mask=None, int axis=0, unsigned num_threads=1):
    """Calculate a median along the `axis`.

    Args:
        data (numpy.ndarray) : Intensity frames.
        mask (numpy.ndarray) : Bad pixel mask.
        axis (int) : Array axis along which median values are calculated.
        num_threads (int) : Number of threads used in the calculations.

    Returns:
        numpy.ndarray : Array of medians along the given axis.
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
        elif type_num == np.NPY_UINT64:
            fail = median_c(_out, _data, _mask, ndim, _dims, 8, axis, compare_ulong, num_threads)
        else:
            raise TypeError('data argument has incompatible type: {:s}'.format(data.dtype))
    if fail:
        raise RuntimeError('C backend exited with error.')

    free(odims)
    return out

def median_filter(np.ndarray data not None, object size=None, np.ndarray footprint=None,
                  np.ndarray mask=None, str mode='reflect', double cval=0.0, unsigned num_threads=1):
    """Calculate a median along the `axis`.

    Args:
        data (numpy.ndarray) : Intensity frames.
        size (Optional[Union[int, Tuple[int, ...]]]) : See footprint, below. Ignored if footprint
            is given.
        footprint (Optional[numpy.ndarray]) :  Either size or footprint must be defined. size
            gives the shape that is taken from the input array, at every element position, to
            define the input to the filter function. footprint is a boolean array that specifies
            (implicitly) a shape, but also which of the elements within this shape will get passed
            to the filter function. Thus size=(n,m) is equivalent to footprint=np.ones((n,m)).
            We adjust size to the number of dimensions of the input array, so that, if the input
            array is shape (10,10,10), and size is 2, then the actual size used is (2,2,2). When
            footprint is given, size is ignored.
        mask (Optional[numpy.ndarray]) : Bad pixel mask.
        mode (str) : The mode parameter determines how the input array is extended when the
            filter overlaps a border. Default value is 'reflect'. The valid values and their
            behavior is as follows:

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
        cval (float) : Value to fill past edges of input if mode is 'constant'. Default is 0.0.
        num_threads (int) : Number of threads used in the calculations.

    Returns:
        numpy.ndarray : Filtered array. Has the same shape as `input`.
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
        elif type_num == np.NPY_UINT64:
            fail = median_filter_c(_out, _data, _mask, ndim, _dims, 8, _fsize, _fmask, _mode, _cval, compare_ulong, num_threads)
        else:
            raise TypeError('data argument has incompatible type: {:s}'.format(data.dtype))
    if fail:
        raise RuntimeError('C backend exited with error.')

    return out

def maximum_filter(np.ndarray data not None, object size=None, np.ndarray footprint=None, np.ndarray mask=None,
                   str mode='reflect', double cval=0.0, int num_threads=1):
    """Calculate a multidimensional maximum filter.

    Parameters
    ----------
        data (numpy.ndarray) : Intensity frames.
        size (Optional[Union[int, Tuple[int, ...]]]) : See footprint, below. Ignored if footprint
            is given.
        footprint (Optional[numpy.ndarray]) :  Either size or footprint must be defined. size
            gives the shape that is taken from the input array, at every element position, to
            define the input to the filter function. footprint is a boolean array that specifies
            (implicitly) a shape, but also which of the elements within this shape will get passed
            to the filter function. Thus size=(n,m) is equivalent to footprint=np.ones((n,m)).
            We adjust size to the number of dimensions of the input array, so that, if the input
            array is shape (10,10,10), and size is 2, then the actual size used is (2,2,2). When
            footprint is given, size is ignored.
        mode (str) : The mode parameter determines how the input array is extended when the
            filter overlaps a border. Default value is 'reflect'. The valid values and their
            behavior is as follows:

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
        cval (float) : Value to fill past edges of input if mode is 'constant'. Default is 0.0.
        num_threads (int) : Number of threads.

    Returns:
        numpy.ndarray : Filtered array. Has the same shape as `input`.
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

    cdef np.npy_intp *dims = [_n_idxs, 4]
    cdef np.ndarray idxs = ArrayWrapper.from_ptr(<void *>_idxs).to_ndarray(2, dims, np.NPY_UINT32)

    return idxs

def subtract_background(data: np.ndarray, mask: np.ndarray, whitefield: np.ndarray,
                        flatfields: np.ndarray=None, num_threads: cython.uint=1,
                        out: np.ndarray=None) -> np.ndarray:
    data = check_array(data, np.NPY_UINT32)
    mask = check_array(mask, np.NPY_BOOL)
    whitefield = check_array(whitefield, np.NPY_FLOAT32)

    cdef int i, j, k
    cdef float res, w0, w1

    if out is None:
        out = <np.ndarray>np.PyArray_SimpleNew(data.ndim, data.shape, np.NPY_FLOAT32)
    else:
        out = check_array(out, np.NPY_FLOAT32)

    cdef np.uint32_t[:, :, ::1] _data = data
    cdef np.npy_bool[:, :, ::1] _mask = mask
    cdef np.float32_t[:, ::1] _whitefield = whitefield
    cdef np.float32_t[:, :, ::1] _out = out
    cdef np.uint32_t[:, :, ::1] _flatfields

    cdef int n_frames = _data.shape[0]
    num_threads = n_frames if <int>num_threads > n_frames else <int>num_threads

    if flatfields is None:
        for i in prange(n_frames, schedule='guided', num_threads=num_threads, nogil=True):
            w0 = 0.0; w1 = 0.0
            for j in range(_data.shape[1]):
                for k in range(_data.shape[2]):
                    if _mask[i, j, k]:
                        w0 = w0 + <float>_data[i, j, k] * _whitefield[j, k]
                        w1 = w1 + _whitefield[j, k] * _whitefield[j, k]
            w0 = w0 / w1 if w1 > 0.0 else 1.0
            for j in range(_data.shape[1]):
                for k in range(_data.shape[2]):
                    if _mask[i, j, k]:
                        res = <float>_data[i, j, k] - w0 * _whitefield[j, k]
                        _out[i, j, k] = res if res > 0.0 else 0.0
                    else:
                        _out[i, j, k] = 0.0

    else:
        flatfields = check_array(flatfields, np.NPY_UINT32)
        _flatfields = flatfields

        for i in prange(n_frames, schedule='guided', num_threads=num_threads, nogil=True):
            w0 = 0.0; w1 = 0.0
            for j in range(_data.shape[1]):
                for k in range(_data.shape[2]):
                    if _mask[i, j, k]:
                        w0 = w0 + <float>_data[i, j, k] * <float>_flatfields[i, j, k]
                        w1 = w1 + <float>_flatfields[i, j, k] * <float>_flatfields[i, j, k]
            w0 = w0 / w1 if w1 > 0.0 else 1.0
            for j in range(_data.shape[1]):
                for k in range(_data.shape[2]):
                    if _mask[i, j, k]:
                        res = <float>_data[i, j, k] - w0 * <float>_flatfields[i, j, k]
                        _out[i, j, k] = res if res > 0.0 else 0.0
                    else:
                        _out[i, j, k] = 0.0

    return out