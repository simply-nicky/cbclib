import numpy as np
from .image_proc cimport check_array, normalize_sequence
from .line_detector cimport ArrayWrapper

cdef loss_func lfuncs[3]
cdef loss_func gfuncs[3]
cdef void build_loss():
    lfuncs[0] = l2_loss
    lfuncs[1] = l1_loss
    lfuncs[2] = huber_loss
    gfuncs[0] = l2_grad
    gfuncs[1] = l1_grad
    gfuncs[2] = huber_grad

cdef dict loss_scheme = {'l2': 0, 'l1': 1, 'huber': 2}

build_loss()

# Numpy must be initialized. When using numpy from C or Cython you must
# *ALWAYS* do that, or you will have segfaults
np.import_array()

def unique_indices(np.ndarray idxs not None):
    idxs = check_array(idxs, np.NPY_UINT32)

    cdef int fail = 0
    cdef unsigned long npts = idxs.size
    cdef unsigned long ipts
    cdef unsigned *_uniq
    cdef unsigned *_iidxs
    cdef unsigned *_idxs = <unsigned *>np.PyArray_DATA(idxs)
    
    cdef np.ndarray inv = np.PyArray_SimpleNew(1, idxs.shape, np.NPY_UINT32)
    cdef unsigned *_inv = <unsigned *>np.PyArray_DATA(inv)

    with nogil:
        fail = unique_idxs(&_uniq, &_iidxs, &ipts, _idxs, _inv, npts)

    if fail:
        raise RuntimeError('C backend exited with error.')

    cdef np.npy_intp *shape = [ipts,]
    cdef np.ndarray uniq = ArrayWrapper.from_ptr(<void *>_uniq).to_ndarray(1, shape, np.NPY_UINT32)
    shape[0] = ipts + 1
    cdef np.ndarray iidxs = ArrayWrapper.from_ptr(<void *>_iidxs).to_ndarray(1, shape, np.NPY_UINT32)
    return uniq, iidxs, inv

def kr_predict(np.ndarray y not None, np.ndarray x not None, np.ndarray x_hat, float sigma,
               np.ndarray w=None, unsigned int num_threads=1):
    y = check_array(y, np.NPY_FLOAT32)
    x = check_array(x, np.NPY_FLOAT32)
    x_hat = check_array(x_hat, np.NPY_FLOAT32)

    cdef int i
    cdef float *_w
    if w is None:
        w = <np.ndarray>np.PyArray_SimpleNew(y.ndim, y.shape, np.NPY_FLOAT32)
        _w = <float *>np.PyArray_DATA(w)
        for i in range(y.size):
            _w[i] = 1.0
    else:
        w = check_array(w, np.NPY_FLOAT32)
        _w = <float *>np.PyArray_DATA(w)

    cdef int fail = 0
    cdef np.npy_intp ndim = x.shape[x.ndim - 1]
    cdef np.npy_intp npts = x.size / ndim, nhat = x_hat.size / ndim

    if x.shape[x.ndim - 1] != x_hat.shape[x_hat.ndim - 1]:
        raise ValueError('`x` and `x_hat` have incompatible shapes')
    if npts != y.size:
        raise ValueError('`x` and `y` have incompatible shapes')

    cdef np.ndarray y_hat = <np.ndarray>np.PyArray_SimpleNew(x_hat.ndim - 1, x_hat.shape, np.NPY_FLOAT32)
    cdef float *_y_hat = <float *>np.PyArray_DATA(y_hat)
    cdef float *_y = <float *>np.PyArray_DATA(y)
    cdef float *_x = <float *>np.PyArray_DATA(x)
    cdef float *_x_hat = <float *>np.PyArray_DATA(x_hat)

    with nogil:
        fail = predict_kerreg(_y, _w, _x, npts, ndim, _y_hat, _x_hat, nhat, rbf,
                              sigma, num_threads)

    if fail:
        raise RuntimeError('C backend exited with error.')

    return y_hat

def kr_grid(np.ndarray y not None, np.ndarray x not None, tuple grid not None, float sigma,
            np.ndarray w=None, bint return_roi=True, unsigned int num_threads=1):
    y = check_array(y, np.NPY_FLOAT32)
    x = check_array(x, np.NPY_FLOAT32)

    cdef int i
    cdef float *_w
    if w is None:
        w = <np.ndarray>np.PyArray_SimpleNew(y.ndim, y.shape, np.NPY_FLOAT32)
        _w = <float *>np.PyArray_DATA(w)
        for i in range(y.size):
            _w[i] = 1.0
    else:
        w = check_array(w, np.NPY_FLOAT32)
        _w = <float *>np.PyArray_DATA(w)

    cdef int fail = 0
    cdef int npts = y.size
    cdef int ndim = x.shape[x.ndim - 1]

    if npts != x.shape[0] or npts != w.size:
        raise ValueError('y and x have incompatible shapes')
    if ndim != len(grid):
        raise ValueError('x and grid have incompatible shapes')

    cdef np.ndarray arr
    cdef unsigned long *_gdims = <unsigned long *>malloc(ndim * sizeof(unsigned long))
    cdef float **_gptrs = <float **>malloc(ndim * sizeof(float *))
    for i in range(ndim):
        arr = check_array(grid[i], np.NPY_FLOAT32)
        _gdims[ndim - 1 - i] = arr.size
        _gptrs[i] = <float *>malloc(arr.size * sizeof(float))
        memcpy(_gptrs[i], np.PyArray_DATA(arr), arr.size * sizeof(float))

    cdef np.npy_intp *roi = <np.npy_intp *>malloc(2 * ndim * sizeof(np.npy_intp))
    cdef float *_y_hat
    cdef float *_y = <float *>np.PyArray_DATA(y)
    cdef float *_x = <float *>np.PyArray_DATA(x)
    cdef unsigned long *_roi = <unsigned long *>roi

    with nogil:
        fail = predict_grid(&_y_hat, _roi, _y, _w, _x, npts, ndim, _gptrs, _gdims, rbf,
                            sigma, num_threads)

    if fail:
        raise RuntimeError('C backend exited with error.')

    for i in range(ndim):
        free(_gptrs[i])
    free(_gptrs); free(_gdims)

    cdef np.npy_intp *shape = <np.npy_intp *>malloc(ndim * sizeof(np.npy_intp))
    for i in range(ndim):
        shape[i] = roi[2 * i + 1] - roi[2 * i]
    cdef np.ndarray y_hat = ArrayWrapper.from_ptr(<void *>_y_hat).to_ndarray(ndim, shape, np.NPY_FLOAT32)
    cdef np.ndarray roi_arr = ArrayWrapper.from_ptr(<void *>_roi).to_ndarray(1, [2 * ndim,], np.NPY_INTP)

    if return_roi:
        return y_hat, roi_arr
    else:
        return y_hat

def binterpolate(np.ndarray data not None, tuple grid not None, np.ndarray coords not None, unsigned num_threads=1):
    data = check_array(data, np.NPY_FLOAT32)
    coords = check_array(coords, np.NPY_FLOAT32)

    cdef int i, fail = 0
    cdef int ndim = data.ndim
    if ndim != len(grid) or ndim != coords.shape[coords.ndim - 1]:
        raise ValueError('data and grid have incompatible shapes')

    cdef unsigned long *dims = <unsigned long *>data.shape
    cdef unsigned long ncrd = coords.size / coords.shape[coords.ndim - 1]

    cdef np.ndarray arr
    cdef float **_gptrs = <float **>malloc(ndim * sizeof(float *))
    for i in range(ndim):
        arr = check_array(grid[i], np.NPY_FLOAT32)
        if data.shape[ndim - 1 - i] != arr.size:
            raise ValueError('data and grid have incompatible shapes')
        _gptrs[i] = <float *>malloc(arr.size * sizeof(float))
        memcpy(_gptrs[i], np.PyArray_DATA(arr), arr.size * sizeof(float))

    cdef np.ndarray out = np.PyArray_SimpleNew(coords.ndim - 1, coords.shape, np.NPY_FLOAT32)
    cdef float *_out = <float *>np.PyArray_DATA(out)
    cdef float *_data = <float *>np.PyArray_DATA(data)
    cdef float *_coords = <float *>np.PyArray_DATA(coords)

    with nogil:
        fail = interp_bi(_out, _data, ndim, dims, _gptrs, _coords, ncrd, num_threads)

    if fail:
        raise RuntimeError('C backend exited with error.')

    for i in range(ndim):
        free(_gptrs[i])
    free(_gptrs)

    return out

def poisson_criterion(np.ndarray x not None, object shape not None, np.ndarray ij not None, np.ndarray I0 not None,
                      np.ndarray bgd not None, np.ndarray xtal_bi not None, np.ndarray prof not None, np.ndarray fidxs not None,
                      np.ndarray idxs not None, np.ndarray hkl_idxs not None, np.ndarray oidxs=None, unsigned num_threads=1):
    if I0.size != bgd.size or I0.size != xtal_bi.size or I0.size != prof.size or \
       I0.size != idxs.size or I0.size != fidxs[fidxs.size - 1]:
        raise ValueError('Input arrays have incompatible sizes')

    cdef np.ndarray _shape = normalize_sequence(shape, 2, np.NPY_INTP)
    cdef unsigned long *_dims = [_shape[0], _shape[1]]

    x = check_array(x, np.NPY_FLOAT32)
    ij = check_array(ij, np.NPY_UINT32)
    I0 = check_array(I0, np.NPY_UINT32)
    bgd = check_array(bgd, np.NPY_FLOAT32)
    xtal_bi = check_array(xtal_bi, np.NPY_FLOAT32)
    prof = check_array(prof, np.NPY_FLOAT32)
    fidxs = check_array(fidxs, np.NPY_UINT32)
    idxs = check_array(idxs, np.NPY_UINT32)
    hkl_idxs = check_array(hkl_idxs, np.NPY_UINT32)

    cdef unsigned long _isize = idxs[idxs.size - 1] + 1
    if x.size < _isize or hkl_idxs.size != _isize:
        raise ValueError("idxs is incompatible with x and hkl_idxs")
    cdef unsigned long _hkl_size = x.size - _isize

    cdef unsigned long _osize
    if oidxs is None:
        oidxs = np.PyArray_ZEROS(1, [_isize,], np.NPY_UINT32, 0)
        _osize = 1
    else:
        _osize = np.PyArray_Max(oidxs, 0, <np.ndarray>NULL) + 1

    cdef np.ndarray out = np.PyArray_ZEROS(1, [_osize,], np.NPY_FLOAT64, 0)
    cdef np.ndarray grad = np.PyArray_ZEROS(x.ndim, x.shape, np.NPY_FLOAT64, 0)
    cdef double *_out = <double *>np.PyArray_DATA(out)
    cdef double *_grad = <double *>np.PyArray_DATA(grad)
    cdef float *_x = <float *>np.PyArray_DATA(x)
    cdef unsigned *_ij = <unsigned *>np.PyArray_DATA(ij)
    cdef unsigned *_I0 = <unsigned *>np.PyArray_DATA(I0)
    cdef float *_bgd = <float *>np.PyArray_DATA(bgd)
    cdef float *_xtal_bi = <float *>np.PyArray_DATA(xtal_bi)
    cdef float *_prof = <float *>np.PyArray_DATA(prof)
    cdef unsigned *_fidxs = <unsigned *>np.PyArray_DATA(fidxs)
    cdef unsigned long _fsize = fidxs.size - 1
    cdef unsigned *_idxs = <unsigned *>np.PyArray_DATA(idxs)
    cdef unsigned *_hkl_idxs = <unsigned *>np.PyArray_DATA(hkl_idxs)
    cdef unsigned *_oidxs = <unsigned *>np.PyArray_DATA(oidxs)
    cdef int fail

    with nogil:
        fail = poisson_likelihood(_out, _grad, _x, _ij, _dims, _I0, _bgd, _xtal_bi, _prof, _fidxs, _fsize, _idxs,
                                  _isize, _hkl_idxs, _hkl_size, _oidxs, _osize, num_threads)

    if fail:
        raise RuntimeError('C backend exited with error.')

    return out, grad

def ls_criterion(np.ndarray x not None, object shape not None, np.ndarray ij not None, np.ndarray I0 not None, np.ndarray bgd not None,
                 np.ndarray xtal_bi not None, np.ndarray prof not None, np.ndarray fidxs not None, np.ndarray idxs not None,
                 np.ndarray hkl_idxs not None, np.ndarray oidxs=None, str loss='l2', unsigned num_threads=1):
    if I0.size != bgd.size or I0.size != xtal_bi.size or I0.size != prof.size or \
       I0.size != idxs.size or I0.size != fidxs[fidxs.size - 1]:
        raise ValueError('Input arrays have incompatible sizes')
    if loss not in loss_scheme:
        raise ValueError(f"Invalid loss keyword: '{loss}'")

    cdef np.ndarray _shape = normalize_sequence(shape, 2, np.NPY_INTP)
    cdef unsigned long *_dims = [_shape[0], _shape[1]]

    x = check_array(x, np.NPY_FLOAT32)
    ij = check_array(ij, np.NPY_UINT32)
    I0 = check_array(I0, np.NPY_UINT32)
    bgd = check_array(bgd, np.NPY_FLOAT32)
    xtal_bi = check_array(xtal_bi, np.NPY_FLOAT32)
    prof = check_array(prof, np.NPY_FLOAT32)
    fidxs = check_array(fidxs, np.NPY_UINT32)
    idxs = check_array(idxs, np.NPY_UINT32)
    hkl_idxs = check_array(hkl_idxs, np.NPY_UINT32)

    cdef unsigned long _isize = idxs[idxs.size - 1] + 1
    if x.size < _isize or hkl_idxs.size != _isize:
        raise ValueError("idxs is incompatible with x and hkl_idxs")
    cdef unsigned long _hkl_size = x.size - _isize

    cdef unsigned long _osize
    if oidxs is None:
        oidxs = np.PyArray_ZEROS(1, [_isize,], np.NPY_UINT32, 0)
        _osize = 1
    else:
        _osize = np.PyArray_Max(oidxs, 0, <np.ndarray>NULL) + 1

    cdef np.ndarray out = np.PyArray_ZEROS(1, [_osize,], np.NPY_FLOAT64, 0)
    cdef np.ndarray grad = np.PyArray_ZEROS(x.ndim, x.shape, np.NPY_FLOAT64, 0)
    cdef double *_out = <double *>np.PyArray_DATA(out)
    cdef double *_grad = <double *>np.PyArray_DATA(grad)
    cdef float *_x = <float *>np.PyArray_DATA(x)
    cdef unsigned *_ij = <unsigned *>np.PyArray_DATA(ij)
    cdef unsigned *_I0 = <unsigned *>np.PyArray_DATA(I0)
    cdef float *_bgd = <float *>np.PyArray_DATA(bgd)
    cdef float *_xtal_bi = <float *>np.PyArray_DATA(xtal_bi)
    cdef float *_prof = <float *>np.PyArray_DATA(prof)
    cdef unsigned *_fidxs = <unsigned *>np.PyArray_DATA(fidxs)
    cdef unsigned long _fsize = fidxs.size - 1
    cdef unsigned *_idxs = <unsigned *>np.PyArray_DATA(idxs)
    cdef unsigned *_hkl_idxs = <unsigned *>np.PyArray_DATA(hkl_idxs)
    cdef unsigned *_oidxs = <unsigned *>np.PyArray_DATA(oidxs)
    cdef loss_func _lfunc = lfuncs[loss_scheme[loss]]
    cdef loss_func _gfunc = gfuncs[loss_scheme[loss]]
    cdef int fail

    with nogil:
        fail = least_squares(_out, _grad, _x, _ij, _dims, _I0, _bgd, _xtal_bi, _prof, _fidxs, _fsize, _idxs,
                             _isize, _hkl_idxs, _hkl_size, _oidxs, _osize, _lfunc, _gfunc, num_threads)

    return out, grad

def unmerge_signal(np.ndarray x not None, object shape not None, np.ndarray ij not None, np.ndarray I0 not None,
                   np.ndarray bgd not None, np.ndarray xtal_bi not None, np.ndarray prof not None, np.ndarray fidxs not None,
                   np.ndarray idxs not None, np.ndarray hkl_idxs not None, unsigned num_threads=1):
    if I0.size != bgd.size or I0.size != xtal_bi.size or I0.size != prof.size or \
       I0.size != idxs.size or I0.size != fidxs[fidxs.size - 1]:
        raise ValueError('Input arrays have incompatible sizes')

    cdef np.ndarray _shape = normalize_sequence(shape, 2, np.NPY_INTP)
    cdef unsigned long *_dims = [_shape[0], _shape[1]]

    x = check_array(x, np.NPY_FLOAT32)
    ij = check_array(ij, np.NPY_UINT32)
    I0 = check_array(I0, np.NPY_UINT32)
    bgd = check_array(bgd, np.NPY_FLOAT32)
    xtal_bi = check_array(xtal_bi, np.NPY_FLOAT32)
    prof = check_array(prof, np.NPY_FLOAT32)
    fidxs = check_array(fidxs, np.NPY_UINT32)
    idxs = check_array(idxs, np.NPY_UINT32)
    hkl_idxs = check_array(hkl_idxs, np.NPY_UINT32)

    cdef unsigned long _isize = idxs[idxs.size - 1] + 1
    if x.size < _isize or hkl_idxs.size != _isize:
        raise ValueError("idxs is incompatible with x and hkl_idxs")
    cdef unsigned long _hkl_size = x.size - _isize

    cdef np.ndarray I_hat = np.PyArray_ZEROS(I0.ndim, I0.shape, np.NPY_FLOAT32, 0)
    cdef float *_I_hat = <float *>np.PyArray_DATA(I_hat)
    cdef float *_x = <float *>np.PyArray_DATA(x)
    cdef unsigned *_ij = <unsigned *>np.PyArray_DATA(ij)
    cdef unsigned *_I0 = <unsigned *>np.PyArray_DATA(I0)
    cdef float *_bgd = <float *>np.PyArray_DATA(bgd)
    cdef float *_xtal_bi = <float *>np.PyArray_DATA(xtal_bi)
    cdef float *_prof = <float *>np.PyArray_DATA(prof)
    cdef unsigned *_fidxs = <unsigned *>np.PyArray_DATA(fidxs)
    cdef unsigned long _fsize = fidxs.size - 1
    cdef unsigned *_idxs = <unsigned *>np.PyArray_DATA(idxs)
    cdef unsigned *_hkl_idxs = <unsigned *>np.PyArray_DATA(hkl_idxs)
    cdef int fail = 0

    with nogil:
        fail = unmerge_sgn(_I_hat, _x, _ij, _dims, _I0, _bgd, _xtal_bi, _prof, _fidxs, _fsize, _idxs,
                           _isize, _hkl_idxs, _hkl_size, num_threads)

    if fail:
        raise RuntimeError('C backend exited with error.')

    return I_hat
