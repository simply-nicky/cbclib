import numpy as np
from cython.parallel import prange
from .line_detector cimport ArrayWrapper

cdef line_profile profiles[4]
cdef void build_profiles():
    profiles[0] = linear_profile
    profiles[1] = quad_profile
    profiles[2] = tophat_profile
    profiles[3] = gauss_profile

cdef dict profile_scheme
profile_scheme = {'linear': 0, 'quad': 1, 'tophat': 2, 'gauss': 3}

# Set the cleanup routine
cdef void _cleanup():
    fftw_cleanup()
    fftw_cleanup_threads()

build_profiles()
fftw_init_threads()

Py_AtExit(_cleanup)

# Numpy must be initialized. When using numpy from C or Cython you must
# *ALWAYS* do that, or you will have segfaults
np.import_array()

def next_fast_len(unsigned target, str backend='numpy'):
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
    cdef int fail = 0
    cdef int ndim = array.ndim
    axis = axis if axis >= 0 else ndim + axis
    if axis >= ndim:
        raise ValueError(f"Axis {axis:d} is out of bounds")

    cdef np.npy_intp ksize = np.PyArray_DIM(kernel, 0)
    cdef int _mode = mode_to_code(mode)
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

def gaussian_kernel(double sigma, unsigned order=0, double truncate=4.0):
    cdef np.npy_intp radius = <np.npy_intp>(sigma * truncate)
    cdef np.npy_intp *dims = [2 * radius + 1,]
    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(1, dims, np.NPY_FLOAT64)
    cdef double *_out = <double *>np.PyArray_DATA(out)
    with nogil:
        gauss_kernel1d(_out, sigma, order, dims[0], 1)
    return out

def gaussian_filter(np.ndarray inp not None, object sigma not None, object order not None=0,
                    str mode='reflect', double cval=0.0, double truncate=4.0, str backend='numpy',
                    unsigned num_threads=1):
    cdef int ndim = inp.ndim
    cdef np.ndarray sigmas = normalise_sequence(sigma, ndim, np.NPY_FLOAT64)
    cdef np.ndarray orders = normalise_sequence(order, ndim, np.NPY_UINT32)
    cdef double *_sig = <double *>np.PyArray_DATA(sigmas)
    cdef unsigned *_ord = <unsigned *>np.PyArray_DATA(orders)

    cdef int n
    for n in range(ndim):
        if inp.shape[n] == 1:
            sigmas[n] = 0.0

    cdef int fail = 0
    cdef int _mode = mode_to_code(mode)
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
    cdef int ndim = inp.ndim
    cdef np.ndarray sigmas = normalise_sequence(sigma, ndim, np.NPY_FLOAT64)
    cdef double *_sig = <double *>np.PyArray_DATA(sigmas)

    cdef int n
    for n in range(ndim):
        if inp.shape[n] == 1:
            sigmas[n] = 0.0

    cdef int fail = 0
    cdef int _mode = mode_to_code(mode)
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

def median(np.ndarray inp not None, np.ndarray mask=None, object axis=0,
           unsigned num_threads=1):
    if mask is None:
        mask = <np.ndarray>np.PyArray_SimpleNew(inp.ndim, inp.shape, np.NPY_BOOL)
        np.PyArray_FILLWBYTE(mask, 1)
    else:
        mask = check_array(mask, np.NPY_BOOL)
        if memcmp(inp.shape, mask.shape, inp.ndim * sizeof(np.npy_intp)):
            raise ValueError('mask and inp arrays must have identical shapes')

    cdef int ndim = 0, i, j, repeats = 1
    cdef np.ndarray axarr = to_array(axis, np.NPY_INT32)
    for i in range(axarr.size):
        axarr[i] = axarr[i] if axarr[i] >= 0 else inp.ndim + axarr[i]
        if axarr[i] >= inp.ndim:
            raise ValueError(f'Axis {axarr[i]:d} is out of bounds')

    for i in range(inp.ndim):
        j = 0
        while j < axarr.size and axarr[j] != i:
            j += 1

        if j == axarr.size:
            inp = np.PyArray_SwapAxes(inp, ndim, i)
            repeats *= inp.shape[ndim]
            ndim += 1

    cdef np.npy_intp *new_dims = <np.npy_intp *>malloc((ndim + 1) * sizeof(np.npy_intp))
    cdef np.npy_intp *odims = <np.npy_intp *>malloc(ndim * sizeof(np.npy_intp))
    for i in range(ndim):
        new_dims[i] = inp.shape[i]
        odims[i] = inp.shape[i]
    new_dims[ndim] = inp.size / repeats
    new_shape = <np.PyArray_Dims *>malloc(sizeof(np.PyArray_Dims))
    new_shape[0].ptr = new_dims; new_shape[0].len = ndim + 1

    cdef int type_num = np.PyArray_TYPE(inp)
    inp = np.PyArray_Newshape(inp, new_shape, np.NPY_CORDER)
    if not np.PyArray_IS_C_CONTIGUOUS(inp):
        inp = np.PyArray_GETCONTIGUOUS(inp)
    mask = np.PyArray_Newshape(mask, new_shape, np.NPY_CORDER)
    if not np.PyArray_IS_C_CONTIGUOUS(mask):
        mask = np.PyArray_GETCONTIGUOUS(mask)
    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(ndim, odims, type_num)
    free(odims); free(new_dims); free(new_shape)

    cdef void *_out = <void *>np.PyArray_DATA(out)
    cdef void *_inp = <void *>np.PyArray_DATA(inp)
    cdef unsigned char *_mask = <unsigned char *>np.PyArray_DATA(mask)
    cdef unsigned long *_dims = <unsigned long *>inp.shape

    with nogil:
        if type_num == np.NPY_FLOAT64:
            fail = median_c(_out, _inp, _mask, ndim + 1, _dims, 8, ndim, compare_double, num_threads)
        elif type_num == np.NPY_FLOAT32:
            fail = median_c(_out, _inp, _mask, ndim + 1, _dims, 4, ndim, compare_float, num_threads)
        elif type_num == np.NPY_INT32:
            fail = median_c(_out, _inp, _mask, ndim + 1, _dims, 4, ndim, compare_int, num_threads)
        elif type_num == np.NPY_UINT32:
            fail = median_c(_out, _inp, _mask, ndim + 1, _dims, 4, ndim, compare_uint, num_threads)
        elif type_num == np.NPY_UINT64:
            fail = median_c(_out, _inp, _mask, ndim + 1, _dims, 8, ndim, compare_ulong, num_threads)
        else:
            raise TypeError(f'inp argument has incompatible type: {str(inp.dtype)}')
    if fail:
        raise RuntimeError('C backend exited with error.')

    return out

def median_filter(np.ndarray inp not None, object size=None, np.ndarray footprint=None,
                  np.ndarray mask=None, np.ndarray inp_mask=None, str mode='reflect', double cval=0.0,
                  unsigned num_threads=1):
    if not np.PyArray_IS_C_CONTIGUOUS(inp):
        inp = np.PyArray_GETCONTIGUOUS(inp)

    cdef int ndim = inp.ndim
    cdef np.npy_intp *dims = inp.shape

    if mask is None:
        mask = <np.ndarray>np.PyArray_SimpleNew(ndim, dims, np.NPY_BOOL)
        np.PyArray_FILLWBYTE(mask, 1)
    else:
        mask = check_array(mask, np.NPY_BOOL)

    if inp_mask is None:
        inp_mask = mask
    else:
        inp_mask = check_array(inp_mask, np.NPY_BOOL)

    if size is None and footprint is None:
        raise ValueError('size or footprint must be provided.')

    cdef unsigned long *_fsize
    cdef np.ndarray fsize
    if size is None:
        _fsize = <unsigned long *>footprint.shape
    else:
        fsize = normalise_sequence(size, ndim, np.NPY_INTP)
        _fsize = <unsigned long *>np.PyArray_DATA(fsize)

    if footprint is None:
        footprint = <np.ndarray>np.PyArray_SimpleNew(ndim, <np.npy_intp *>_fsize, np.NPY_BOOL)
        np.PyArray_FILLWBYTE(footprint, 1)
    else:
        footprint = check_array(footprint, np.NPY_BOOL)

    if footprint.ndim != ndim:
        raise ValueError('footprint and size must have the same number of dimensions as the input')
    cdef unsigned char *_fmask = <unsigned char *>np.PyArray_DATA(footprint)

    cdef unsigned long *_dims = <unsigned long *>dims
    cdef int type_num = np.PyArray_TYPE(inp)
    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(ndim, dims, type_num)
    cdef void *_out = <void *>np.PyArray_DATA(out)
    cdef void *_inp = <void *>np.PyArray_DATA(inp)
    cdef unsigned char *_mask = <unsigned char *>np.PyArray_DATA(mask)
    cdef unsigned char *_imask = <unsigned char *>np.PyArray_DATA(inp_mask)
    cdef int _mode = mode_to_code(mode)
    cdef void *_cval = <void *>&cval

    with nogil:
        if type_num == np.NPY_FLOAT64:
            fail = median_filter_c(_out, _inp, _mask, _imask, ndim, _dims, 8, _fsize, _fmask, _mode, _cval, compare_double, num_threads)
        elif type_num == np.NPY_FLOAT32:
            fail = median_filter_c(_out, _inp, _mask, _imask, ndim, _dims, 4, _fsize, _fmask, _mode, _cval, compare_float, num_threads)
        elif type_num == np.NPY_INT32:
            fail = median_filter_c(_out, _inp, _mask, _imask, ndim, _dims, 4, _fsize, _fmask, _mode, _cval, compare_int, num_threads)
        elif type_num == np.NPY_UINT32:
            fail = median_filter_c(_out, _inp, _mask, _imask, ndim, _dims, 4, _fsize, _fmask, _mode, _cval, compare_uint, num_threads)
        elif type_num == np.NPY_UINT64:
            fail = median_filter_c(_out, _inp, _mask, _imask, ndim, _dims, 8, _fsize, _fmask, _mode, _cval, compare_ulong, num_threads)
        else:
            raise TypeError(f'inp argument has incompatible type: {str(inp.dtype)}')
    if fail:
        raise RuntimeError('C backend exited with error.')

    return out

def maximum_filter(np.ndarray inp not None, object size=None, np.ndarray footprint=None,
                   np.ndarray mask=None, np.ndarray inp_mask=None,  str mode='reflect', double cval=0.0,
                   int num_threads=1):
    if not np.PyArray_IS_C_CONTIGUOUS(inp):
        inp = np.PyArray_GETCONTIGUOUS(inp)

    cdef int ndim = inp.ndim
    cdef np.npy_intp *dims = inp.shape

    if mask is None:
        mask = <np.ndarray>np.PyArray_SimpleNew(ndim, dims, np.NPY_BOOL)
        np.PyArray_FILLWBYTE(mask, 1)
    else:
        mask = check_array(mask, np.NPY_BOOL)

    if inp_mask is None:
        inp_mask = mask
    else:
        inp_mask = check_array(inp_mask, np.NPY_BOOL)

    if size is None and footprint is None:
        raise ValueError('size or footprint must be provided.')

    cdef unsigned long *_fsize
    cdef np.ndarray fsize
    if size is None:
        _fsize = <unsigned long *>footprint.shape
    else:
        fsize = normalise_sequence(size, ndim, np.NPY_INTP)
        _fsize = <unsigned long *>np.PyArray_DATA(fsize)

    if footprint is None:
        footprint = <np.ndarray>np.PyArray_SimpleNew(ndim, <np.npy_intp *>_fsize, np.NPY_BOOL)
        np.PyArray_FILLWBYTE(footprint, 1)
    else:
        footprint = check_array(footprint, np.NPY_BOOL)

    if footprint.ndim != ndim:
        raise ValueError('footprint and size must have the same number of dimensions as the input')
    cdef unsigned char *_fmask = <unsigned char *>np.PyArray_DATA(footprint)

    cdef unsigned long *_dims = <unsigned long *>dims
    cdef int type_num = np.PyArray_TYPE(inp)
    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(ndim, dims, type_num)
    cdef void *_out = <void *>np.PyArray_DATA(out)
    cdef void *_inp = <void *>np.PyArray_DATA(inp)
    cdef unsigned char *_mask = <unsigned char *>np.PyArray_DATA(mask)
    cdef unsigned char *_imask = <unsigned char *>np.PyArray_DATA(inp_mask)
    cdef int _mode = mode_to_code(mode)
    cdef void *_cval = <void *>&cval

    with nogil:
        if type_num == np.NPY_FLOAT64:
            fail = maximum_filter_c(_out, _inp, _mask, _imask, ndim, _dims, 8, _fsize, _fmask, _mode, _cval, compare_double, num_threads)
        elif type_num == np.NPY_FLOAT32:
            fail = maximum_filter_c(_out, _inp, _mask, _imask, ndim, _dims, 4, _fsize, _fmask, _mode, _cval, compare_float, num_threads)
        elif type_num == np.NPY_INT32:
            fail = maximum_filter_c(_out, _inp, _mask, _imask, ndim, _dims, 4, _fsize, _fmask, _mode, _cval, compare_int, num_threads)
        elif type_num == np.NPY_UINT32:
            fail = maximum_filter_c(_out, _inp, _mask, _imask, ndim, _dims, 4, _fsize, _fmask, _mode, _cval, compare_uint, num_threads)
        else:
            raise TypeError('inp argument has incompatible type: {:s}'.format(str(inp.dtype)))

    if fail:
        raise RuntimeError('C backend exited with error.')
    return out

def robust_mean(np.ndarray inp not None, object axis=0, double r0=0.0, double r1=0.5,
                int n_iter=12, double lm=9.0, unsigned num_threads=1):
    cdef int ndim = 0, i, j, repeats = 1
    cdef np.ndarray axarr = to_array(axis, np.NPY_INT32)
    for i in range(axarr.size):
        axarr[i] = axarr[i] if axarr[i] >= 0 else inp.ndim + axarr[i]
        if axarr[i] >= inp.ndim:
            raise ValueError(f'Axis {axarr[i]:d} is out of bounds')

    for i in range(inp.ndim):
        j = 0
        while j < axarr.size and axarr[j] != i:
            j += 1

        if j == axarr.size:
            inp = np.PyArray_SwapAxes(inp, ndim, i)
            repeats *= inp.shape[ndim]
            ndim += 1

    cdef np.npy_intp *new_dims = <np.npy_intp *>malloc((ndim + 1) * sizeof(np.npy_intp))
    cdef np.npy_intp *odims = <np.npy_intp *>malloc(ndim * sizeof(np.npy_intp))
    for i in range(ndim):
        new_dims[i] = inp.shape[i]
        odims[i] = inp.shape[i]
    new_dims[ndim] = inp.size / repeats
    new_shape = <np.PyArray_Dims *>malloc(sizeof(np.PyArray_Dims))
    new_shape[0].ptr = new_dims; new_shape[0].len = ndim + 1

    inp = np.PyArray_Newshape(inp, new_shape, np.NPY_CORDER)
    if not np.PyArray_IS_C_CONTIGUOUS(inp):
        inp = np.PyArray_GETCONTIGUOUS(inp)
    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(ndim, odims, np.NPY_FLOAT32)
    free(odims); free(new_dims); free(new_shape)

    cdef int type_num = np.PyArray_TYPE(inp)
    cdef float *_out = <float *>np.PyArray_DATA(out)
    cdef void *_inp = <void *>np.PyArray_DATA(inp)
    cdef unsigned long *_dims = <unsigned long *>inp.shape

    with nogil:
        if type_num == np.NPY_FLOAT64:
            fail = robust_mean_c(_out, _inp, ndim + 1, _dims, 8, ndim, compare_double,
                                 get_double, r0, r1, n_iter, lm, num_threads)
        elif type_num == np.NPY_FLOAT32:
            fail = robust_mean_c(_out, _inp, ndim + 1, _dims, 4, ndim, compare_float,
                                 get_float, r0, r1, n_iter, lm, num_threads)
        elif type_num == np.NPY_INT32:
            fail = robust_mean_c(_out, _inp, ndim + 1, _dims, 4, ndim, compare_int,
                                 get_int, r0, r1, n_iter, lm, num_threads)
        elif type_num == np.NPY_UINT32:
            fail = robust_mean_c(_out, _inp, ndim + 1, _dims, 4, ndim, compare_uint,
                                 get_uint, r0, r1, n_iter, lm, num_threads)
        elif type_num == np.NPY_UINT64:
            fail = robust_mean_c(_out, _inp, ndim + 1, _dims, 8, ndim, compare_ulong,
                                 get_ulong, r0, r1, n_iter, lm, num_threads)
        else:
            raise TypeError(f'inp argument has incompatible type: {str(inp.dtype)}')
    if fail:
        raise RuntimeError('C backend exited with error.')

    return out

def robust_lsq(np.ndarray W not None, np.ndarray y not None, object axis=-1, double r0=0.0,
               double r1=0.5, int n_iter=12, double lm=9.0, unsigned num_threads=1):
    W = check_array(W, np.NPY_FLOAT32)

    cdef int ndim = 0, i, j, repeats = 1
    cdef np.ndarray axarr = to_array(axis, np.NPY_INT32)
    for i in range(axarr.size):
        axarr[i] = axarr[i] if axarr[i] >= 0 else y.ndim + axarr[i]
        if axarr[i] >= y.ndim:
            raise ValueError(f'Axis {axarr[i]:d} is out of bounds')

    for i in range(y.ndim):
        j = 0
        while j < axarr.size and axarr[j] != i:
            j += 1

        if j == axarr.size:
            y = np.PyArray_SwapAxes(y, ndim, i)
            repeats *= y.shape[ndim]
            ndim += 1

    cdef np.npy_intp *new_dims = <np.npy_intp *>malloc((ndim + 1) * sizeof(np.npy_intp))
    cdef np.npy_intp *odims = <np.npy_intp *>malloc((ndim + 1) * sizeof(np.npy_intp))
    for i in range(ndim):
        new_dims[i] = y.shape[i]
        odims[i] = y.shape[i]
    new_dims[ndim] = y.size / repeats
    new_shape = <np.PyArray_Dims *>malloc(sizeof(np.PyArray_Dims))
    new_shape[0].ptr = new_dims; new_shape[0].len = ndim + 1

    cdef int nf = W.size / new_dims[ndim]
    odims[ndim] = nf
    y = np.PyArray_Newshape(y, new_shape, np.NPY_CORDER)
    if not np.PyArray_IS_C_CONTIGUOUS(y):
        y = np.PyArray_GETCONTIGUOUS(y)
    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(ndim + 1, odims, np.NPY_FLOAT32)
    free(odims); free(new_dims); free(new_shape)

    cdef int type_num = np.PyArray_TYPE(y)
    cdef float *_out = <float *>np.PyArray_DATA(out)
    cdef float *_W = <float *>np.PyArray_DATA(W)
    cdef void *_y = <void *>np.PyArray_DATA(y)
    cdef unsigned long *_ydims = <unsigned long *>y.shape

    with nogil:
        if type_num == np.NPY_FLOAT64:
            fail = robust_fit(_out, _W, _y, nf, ndim + 1, _ydims, 8, compare_double,
                              get_double, r0, r1, n_iter, lm, num_threads)
        elif type_num == np.NPY_FLOAT32:
            fail = robust_fit(_out, _W, _y, nf, ndim + 1, _ydims, 4, compare_float,
                              get_float, r0, r1, n_iter, lm, num_threads)
        elif type_num == np.NPY_INT32:
            fail = robust_fit(_out, _W, _y, nf, ndim + 1, _ydims, 4, compare_int,
                              get_int, r0, r1, n_iter, lm, num_threads)
        elif type_num == np.NPY_UINT32:
            fail = robust_fit(_out, _W, _y, nf, ndim + 1, _ydims, 4, compare_uint,
                              get_uint, r0, r1, n_iter, lm, num_threads)
        elif type_num == np.NPY_UINT64:
            fail = robust_fit(_out, _W, _y, nf, ndim + 1, _ydims, 8, compare_ulong,
                              get_ulong, r0, r1, n_iter, lm, num_threads)
        else:
            raise TypeError(f'y argument has incompatible type: {str(y.dtype)}')
    if fail:
        raise RuntimeError('C backend exited with error.')

    return out

def draw_line_mask(object shape not None, object lines not None, int max_val=1, double dilation=0.0,
                   str profile='tophat', unsigned int num_threads=1):
    if profile not in profile_scheme:
        raise ValueError(f"Invalid profile keyword: '{profile}'")

    cdef int ndim = len(shape)
    if ndim < 2:
        raise ValueError(f"Invalid shape: '{shape}'")

    cdef np.ndarray _shape = normalise_sequence(shape, ndim, np.NPY_INTP)
    cdef np.ndarray inp = np.PyArray_ZEROS(ndim, <np.npy_intp *>np.PyArray_DATA(_shape), np.NPY_UINT32, 0)
    cdef unsigned *_inp = <unsigned *>np.PyArray_DATA(inp)

    cdef unsigned long *_dims = <unsigned long *>inp.shape + ndim - 2
    cdef int repeats = inp.size / _dims[0] / _dims[1]

    cdef int i, N
    if ndim == 2:
        N = 1
    else:
        N = len(lines)

    cdef np.ndarray arr
    cdef float **_lptrs = <float **>malloc(N * sizeof(float *))
    cdef unsigned long *_ldims = <unsigned long *>malloc(2 * N * sizeof(unsigned long))

    if N == 1:
        arr = check_array(lines, np.NPY_FLOAT32)
        if arr.ndim != 2 or arr.shape[1] < 5:
            raise ValueError("lines array has an incompatible shape")

        _ldims[0] = arr.shape[0]; _ldims[1] =  arr.shape[1]
        _lptrs[0] = <float *>np.PyArray_DATA(arr)
    else:
        for i in range(N):
            arr = check_array(lines[i], np.NPY_FLOAT32)
            if arr.ndim != 2 or arr.shape[1] < 5:
                raise ValueError("lines array has an incompatible shape")

            _ldims[2 * i] = arr.shape[0]; _ldims[2 * i + 1] = arr.shape[1]
            _lptrs[i] = <float *>malloc(arr.size * sizeof(float))
            memcpy(_lptrs[i], np.PyArray_DATA(arr), arr.size * sizeof(float))

    cdef line_profile _prof = profiles[profile_scheme[profile]]

    if N < repeats:
        repeats = N
    num_threads = repeats if <int>num_threads > repeats else <int>num_threads
    cdef int fail

    for i in prange(repeats, schedule='guided', num_threads=num_threads, nogil=True):
        fail = draw_line_int(_inp + i * _dims[0] * _dims[1], _dims, max_val, _lptrs[i], _ldims + 2 * i,
                             <float>dilation, _prof)

        if fail:
            with gil:
                raise RuntimeError('C backend exited with error.')

    if N > 1:
        for i in range(N):
            free(_lptrs[i])
    free(_lptrs); free(_ldims)

    return inp

def draw_line_image(object shape not None, object lines not None, double dilation=0.0,
                    str profile='gauss', unsigned int num_threads=1):
    if profile not in profile_scheme:
        raise ValueError(f"Invalid profile keyword: '{profile}'")

    cdef int ndim = len(shape)
    if ndim < 2:
        raise ValueError(f"Invalid shape: '{shape}'")

    cdef np.ndarray _shape = normalise_sequence(shape, ndim, np.NPY_INTP)
    cdef np.ndarray inp = np.PyArray_ZEROS(ndim, <np.npy_intp *>np.PyArray_DATA(_shape), np.NPY_FLOAT32, 0)
    cdef float *_inp = <float *>np.PyArray_DATA(inp)

    cdef unsigned long *_dims = <unsigned long *>inp.shape + ndim - 2
    cdef int repeats = inp.size / _dims[0] / _dims[1]

    cdef int i, N
    if ndim == 2:
        N = 1
    else:
        N = len(lines)

    cdef np.ndarray arr
    cdef float **_lptrs = <float **>malloc(N * sizeof(float *))
    cdef unsigned long *_ldims = <unsigned long *>malloc(2 * N * sizeof(unsigned long))

    if N == 1:
        arr = check_array(lines, np.NPY_FLOAT32)
        if arr.ndim != 2 or arr.shape[1] < 5:
            raise ValueError("lines array has an incompatible shape")

        _ldims[0] = arr.shape[0]; _ldims[1] =  arr.shape[1]
        _lptrs[0] = <float *>np.PyArray_DATA(arr)
    else:
        for i in range(N):
            arr = check_array(lines[i], np.NPY_FLOAT32)
            if arr.ndim != 2 or arr.shape[1] < 5:
                raise ValueError("lines array has an incompatible shape")

            _ldims[2 * i] = arr.shape[0]; _ldims[2 * i + 1] = arr.shape[1]
            _lptrs[i] = <float *>malloc(arr.size * sizeof(float))
            memcpy(_lptrs[i], np.PyArray_DATA(arr), arr.size * sizeof(float))

    cdef line_profile _prof = profiles[profile_scheme[profile]]

    if N < repeats:
        repeats = N
    num_threads = repeats if <int>num_threads > repeats else <int>num_threads
    cdef int fail

    for i in prange(repeats, schedule='guided', num_threads=num_threads, nogil=True):
        fail = draw_line_float(_inp + i * _dims[0] * _dims[1], _dims, _lptrs[i], _ldims + 2 * i,
                               <float>dilation, _prof)

        if fail:
            with gil:
                raise RuntimeError('C backend exited with error.')

    if N > 1:
        for i in range(N):
            free(_lptrs[i])
    free(_lptrs); free(_ldims)

    return inp

def draw_line_table(np.ndarray lines not None, object shape=None, double dilation=0.0, str profile='gauss'):
    lines = check_array(lines, np.NPY_FLOAT32)

    if lines.ndim != 2 or lines.shape[1] < 5:
        raise ValueError(f"lines array has an incompatible shape")
    if profile not in profile_scheme:
        raise ValueError(f"Invalid profile keyword: '{profile}'")

    cdef np.ndarray _shape
    if shape is None:
        _shape = np.PyArray_Max(lines, 0, <np.ndarray>NULL)
        _shape[0] += 1; _shape[1] += 1
    else:
        _shape = normalise_sequence(shape, 2, np.NPY_INTP)
    cdef unsigned long *_dims = [_shape[0], _shape[1]]

    cdef unsigned *_idx
    cdef unsigned *_x
    cdef unsigned *_y
    cdef float *_val
    cdef unsigned long _n_idxs
    cdef float *_lines = <float *>np.PyArray_DATA(lines)
    cdef unsigned long *_ldims = <unsigned long *>lines.shape

    cdef line_profile _prof = profiles[profile_scheme[profile]]
    cdef int fail = 0

    with nogil:
        fail = draw_line_index(&_idx, &_x, &_y, &_val, &_n_idxs, _dims, _lines, _ldims, <float>dilation, _prof)
    if fail:
        raise RuntimeError('C backend exited with error.')    

    cdef np.ndarray idx = ArrayWrapper.from_ptr(<void *>_idx).to_ndarray(1, [_n_idxs,], np.NPY_UINT32)
    cdef np.ndarray x = ArrayWrapper.from_ptr(<void *>_x).to_ndarray(1, [_n_idxs,], np.NPY_UINT32)
    cdef np.ndarray y = ArrayWrapper.from_ptr(<void *>_y).to_ndarray(1, [_n_idxs,], np.NPY_UINT32)
    cdef np.ndarray val = ArrayWrapper.from_ptr(<void *>_val).to_ndarray(1, [_n_idxs,], np.NPY_FLOAT32)

    return idx, x, y, val

def normalise_pattern(np.ndarray inp not None, object lines not None, object dilations not None,
                      str profile='tophat', unsigned int num_threads=1):
    if inp.ndim != 3:
        raise ValueError('Input array must be a 3D array.')
    inp = check_array(inp, np.NPY_FLOAT32)

    cdef np.ndarray dils = normalise_sequence(dilations, 3, np.NPY_FLOAT32)
    cdef float *_dils = <float *>np.PyArray_DATA(dils)

    if profile not in profile_scheme:
        raise ValueError(f"Invalid profile keyword: '{profile}'")

    cdef float *_inp = <float *>np.PyArray_DATA(inp)
    cdef unsigned long *_dims = <unsigned long *>inp.shape + 1
    cdef int repeats = inp.size / _dims[0] / _dims[1]

    cdef np.ndarray out = np.PyArray_SimpleNew(3, inp.shape, np.NPY_FLOAT32)
    cdef float *_out = <float *>np.PyArray_DATA(out)

    cdef int i, N = len(lines)
    cdef np.ndarray arr
    cdef float **_lptrs = <float **>malloc(N * sizeof(float *))
    cdef unsigned long *_ldims = <unsigned long *>malloc(2 * N * sizeof(unsigned long))
    for i in range(N):
        arr = check_array(lines[i], np.NPY_FLOAT32)
        if arr.ndim != 2 or arr.shape[1] < 5:
            raise ValueError("lines array has an incompatible shape")
        _ldims[2 * i] = arr.shape[0]; _ldims[2 * i + 1] = arr.shape[1]
        _lptrs[i] = <float *>malloc(arr.size * sizeof(float))
        memcpy(_lptrs[i], np.PyArray_DATA(arr), arr.size * sizeof(float))

    cdef line_profile _prof = profiles[profile_scheme[profile]]

    if N < repeats:
        repeats = N
    num_threads = repeats if <int>num_threads > repeats else <int>num_threads
    cdef int fail

    for i in prange(repeats, schedule='guided', num_threads=num_threads, nogil=True):
        fail = normalise_line(_out + i * _dims[0] * _dims[1], _inp + i * _dims[0] * _dims[1], _dims,
                              _lptrs[i], _ldims + 2 * i, _dils, _prof)

        if fail:
            with gil:
                raise RuntimeError('C backend exited with error.')

    for i in range(N):
        free(_lptrs[i])
    free(_lptrs); free(_ldims)

    return out

def refine_pattern(np.ndarray inp not None, object lines not None, float dilation,
                   str profile='tophat', unsigned int num_threads=1):
    if inp.ndim != 3:
        raise ValueError('Input array must be a 3D array.')
    inp = check_array(inp, np.NPY_FLOAT32)

    if profile not in profile_scheme:
        raise ValueError(f"Invalid profile keyword: '{profile}'")

    cdef float *_inp = <float *>np.PyArray_DATA(inp)
    cdef unsigned long *_dims = <unsigned long *>inp.shape + 1
    cdef int repeats = inp.size / _dims[0] / _dims[1]

    cdef np.ndarray out = np.PyArray_SimpleNew(3, inp.shape, np.NPY_FLOAT32)
    cdef float *_out = <float *>np.PyArray_DATA(out)

    cdef int i, N = len(lines)
    cdef np.ndarray arr
    cdef float **_lptrs = <float **>malloc(N * sizeof(float *))
    cdef unsigned long *_ldims = <unsigned long *>malloc(2 * N * sizeof(unsigned long))
    for i in range(N):
        arr = check_array(lines[i], np.NPY_FLOAT32)
        if arr.ndim != 2 or arr.shape[1] < 5:
            raise ValueError("lines array has an incompatible shape")
        _ldims[2 * i] = arr.shape[0]; _ldims[2 * i + 1] = arr.shape[1]
        _lptrs[i] = <float *>malloc(arr.size * sizeof(float))
        memcpy(_lptrs[i], np.PyArray_DATA(arr), arr.size * sizeof(float))

    cdef line_profile _prof = profiles[profile_scheme[profile]]

    if N < repeats:
        repeats = N
    num_threads = repeats if <int>num_threads > repeats else <int>num_threads
    cdef int fail

    for i in prange(repeats, schedule='guided', num_threads=num_threads, nogil=True):
        fail = refine_line(_lptrs[i], _inp + i * _dims[0] * _dims[1], _dims,
                           _lptrs[i], _ldims + 2 * i, dilation, _prof)

        if fail:
            with gil:
                raise RuntimeError('C backend exited with error.')

    cdef dict line_dict = {}
    for i in range(repeats):
        arr = ArrayWrapper.from_ptr(<void *>_lptrs[i]).to_ndarray(2, <np.npy_intp *>(_ldims + 2 * i), np.NPY_FLOAT32)
        line_dict[i] = arr

    free(_lptrs); free(_ldims)

    return line_dict

def ce_criterion(np.ndarray ij not None, np.ndarray p not None, np.ndarray fidxs not None, object shape not None,
                 object lines not None, double dilation=0.0, double epsilon=1e-12, str profile='gauss',
                 unsigned int num_threads=1):
    if profile not in profile_scheme:
        raise ValueError(f"Invalid profile keyword: '{profile}'")
    if len(shape) != 2:
        raise ValueError(f"Invalid shape: '{shape}'")

    ij = check_array(ij, np.NPY_UINT32)
    p = check_array(p, np.NPY_FLOAT32)
    fidxs = check_array(fidxs, np.NPY_UINT32)

    cdef unsigned long *_dims = [shape[0], shape[1]]
    cdef unsigned *_ij = <unsigned *>np.PyArray_DATA(ij)
    cdef float *_p = <float *>np.PyArray_DATA(p)   
    cdef unsigned *_fidxs = <unsigned *>np.PyArray_DATA(fidxs) 

    cdef int i, N = len(lines)
    cdef np.ndarray arr
    cdef float **_lptrs = <float **>malloc(N * sizeof(float *))
    cdef unsigned long *_ldims = <unsigned long *>malloc(2 * N * sizeof(unsigned long))
    for i in range(N):
        arr = check_array(lines[i], np.NPY_FLOAT32)
        if arr.ndim != 2 or arr.shape[1] < 5:
            raise ValueError("lines array has an incompatible shape")
        _ldims[2 * i] = arr.shape[0]; _ldims[2 * i + 1] = arr.shape[1]
        _lptrs[i] = <float *>malloc(arr.size * sizeof(float))
        memcpy(_lptrs[i], np.PyArray_DATA(arr), arr.size * sizeof(float))

    cdef line_profile _prof = profiles[profile_scheme[profile]]
    cdef double crit

    with nogil:
        crit = cross_entropy(_ij, _p, _fidxs, _dims, _lptrs, _ldims, N, dilation, epsilon, _prof, num_threads)

    for i in range(N):
        free(_lptrs[i])
    free(_lptrs); free(_ldims)

    return crit
