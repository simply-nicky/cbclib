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

def unique_indices(np.ndarray frames not None, np.ndarray indices not None):
    frames = check_array(frames, np.NPY_UINT32)
    indices = check_array(indices, np.NPY_UINT32)

    cdef int fail = 0
    cdef unsigned long npts = frames.size
    cdef unsigned long fpts, ipts
    cdef unsigned *_funiq
    cdef unsigned *_fidxs
    cdef unsigned *_iidxs
    cdef unsigned *_frames = <unsigned *>np.PyArray_DATA(frames)
    cdef unsigned *_indices = <unsigned *>np.PyArray_DATA(indices)

    with nogil:
        fail = unique_indices_c(&_funiq, &_fidxs, &fpts, &_iidxs, &ipts, _frames, _indices, npts)

    if fail:
        raise RuntimeError('C backend exited with error.')

    cdef np.npy_intp *shape = [fpts,]
    cdef np.ndarray funiq = ArrayWrapper.from_ptr(<void *>_funiq).to_ndarray(1, shape, np.NPY_UINT32)
    shape[0] = fpts + 1
    cdef np.ndarray fidxs = ArrayWrapper.from_ptr(<void *>_fidxs).to_ndarray(1, shape, np.NPY_UINT32)
    shape[0] = ipts
    cdef np.ndarray iidxs = ArrayWrapper.from_ptr(<void *>_iidxs).to_ndarray(1, shape, np.NPY_UINT32)
    return funiq, fidxs, iidxs

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

def poisson_criterion(np.ndarray x not None, np.ndarray prof not None, np.ndarray I0 not None, np.ndarray bgd not None,
                      np.ndarray xtal_bi not None, np.ndarray hkl_idxs not None, np.ndarray iidxs not None,
                      unsigned num_threads=1):
    if I0.size != prof.size or I0.size != bgd.size or I0.size != xtal_bi.size or I0.size != iidxs[iidxs.size - 1]:
        raise ValueError('Input arrays have incompatible sizes')

    x = check_array(x, np.NPY_FLOAT32)
    prof = check_array(prof, np.NPY_FLOAT32)
    I0 = check_array(I0, np.NPY_UINT32)
    bgd = check_array(bgd, np.NPY_FLOAT32)
    xtal_bi = check_array(xtal_bi, np.NPY_FLOAT32)
    hkl_idxs = check_array(hkl_idxs, np.NPY_UINT32)
    iidxs = check_array(iidxs, np.NPY_UINT32)

    cdef unsigned long _isize = iidxs.size - 1
    cdef unsigned long _xsize = x.size
    cdef np.ndarray grad = np.PyArray_ZEROS(x.ndim, x.shape, np.NPY_FLOAT32, 0)
    cdef float *_grad = <float *>np.PyArray_DATA(grad)
    cdef float *_x = <float *>np.PyArray_DATA(x)
    cdef float *_prof = <float *>np.PyArray_DATA(prof)
    cdef unsigned *_I0 = <unsigned *>np.PyArray_DATA(I0)
    cdef float *_bgd = <float *>np.PyArray_DATA(bgd)
    cdef float *_xtal_bi = <float *>np.PyArray_DATA(xtal_bi)
    cdef unsigned *_hkl_idxs = <unsigned *>np.PyArray_DATA(hkl_idxs)
    cdef unsigned *_iidxs = <unsigned *>np.PyArray_DATA(iidxs)
    cdef float criterion

    with nogil:
        criterion = poisson_likelihood(_grad, _x, _xsize, _prof, _I0, _bgd, _xtal_bi, _hkl_idxs, _iidxs, _isize,
                                       num_threads)

    return criterion, grad

def ls_criterion(np.ndarray x not None, np.ndarray prof not None, np.ndarray I0 not None, np.ndarray bgd not None,
                 np.ndarray xtal_bi not None, np.ndarray hkl_idxs not None, np.ndarray iidxs not None, str loss='l2',
                 unsigned num_threads=1):
    if I0.size != prof.size or I0.size != bgd.size or I0.size != xtal_bi.size or I0.size != iidxs[iidxs.size - 1]:
        raise ValueError('Input arrays have incompatible sizes')
    if loss not in loss_scheme:
        raise ValueError(f"Invalid loss keyword: '{loss}'")

    x = check_array(x, np.NPY_FLOAT32)
    prof = check_array(prof, np.NPY_FLOAT32)
    I0 = check_array(I0, np.NPY_UINT32)
    bgd = check_array(bgd, np.NPY_FLOAT32)
    xtal_bi = check_array(xtal_bi, np.NPY_FLOAT32)
    hkl_idxs = check_array(hkl_idxs, np.NPY_UINT32)
    iidxs = check_array(iidxs, np.NPY_UINT32)

    cdef unsigned long _isize = iidxs.size - 1
    cdef unsigned long _xsize = x.size
    cdef np.ndarray grad = np.PyArray_ZEROS(x.ndim, x.shape, np.NPY_FLOAT32, 0)
    cdef float *_grad = <float *>np.PyArray_DATA(grad)
    cdef float *_x = <float *>np.PyArray_DATA(x)
    cdef float *_prof = <float *>np.PyArray_DATA(prof)
    cdef unsigned *_I0 = <unsigned *>np.PyArray_DATA(I0)
    cdef float *_bgd = <float *>np.PyArray_DATA(bgd)
    cdef float *_xtal_bi = <float *>np.PyArray_DATA(xtal_bi)
    cdef unsigned *_hkl_idxs = <unsigned *>np.PyArray_DATA(hkl_idxs)
    cdef unsigned *_iidxs = <unsigned *>np.PyArray_DATA(iidxs)
    cdef loss_func _lfunc = lfuncs[loss_scheme[loss]]
    cdef loss_func _gfunc = gfuncs[loss_scheme[loss]]
    cdef float criterion

    with nogil:
        criterion = least_squares(_grad, _x, _xsize, _prof, _I0, _bgd, _xtal_bi, _hkl_idxs, _iidxs, _isize,
                                  _lfunc, _gfunc, num_threads)

    return criterion, grad

def model_fit(np.ndarray x not None, np.ndarray hkl_idxs not None, np.ndarray iidxs not None):
    x = check_array(x, np.NPY_FLOAT32)
    hkl_idxs = check_array(hkl_idxs, np.NPY_UINT32)
    iidxs = check_array(iidxs, np.NPY_UINT32)

    cdef int isize = iidxs.size - 1
    cdef int xsize = x.size

    cdef np.ndarray sfac = np.PyArray_SimpleNew(1, [iidxs[isize],], np.NPY_FLOAT32)
    cdef np.ndarray intercept = np.PyArray_SimpleNew(1, [iidxs[isize],], np.NPY_FLOAT32)

    cdef int i, j
    cdef float sf
    cdef np.float32_t[::1] _sfac = sfac
    cdef np.float32_t[::1] _intercept = intercept
    cdef np.float32_t[::1] _x = x
    cdef np.uint32_t[::1] _hkl_idxs = hkl_idxs
    cdef np.uint32_t[::1] _iidxs = iidxs

    with nogil:
        for i in range(isize):
            sf = exp(_x[isize + _hkl_idxs[i]])
            for j in range(<int>_iidxs[i], <int>_iidxs[i + 1]):
                _intercept[j] = _x[i]; _sfac[j] = sf

    return intercept, sfac
