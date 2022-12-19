import numpy as np
from libc.math cimport sqrt, atan, atan2, sin, cos, floor, ceil
from cython.parallel import prange

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

def find_kins(np.ndarray x not None, np.ndarray y not None, np.ndarray hkl not None, np.ndarray fidxs not None,
              np.ndarray smp_pos not None, np.ndarray rot_mat not None, np.ndarray basis not None,
              double x_pixel_size, double y_pixel_size, unsigned num_threads=1):
    x = check_array(x, np.NPY_UINT32)
    y = check_array(y, np.NPY_UINT32)
    hkl = check_array(hkl, np.NPY_INT64)
    fidxs = check_array(fidxs, np.NPY_UINT32)
    smp_pos = check_array(smp_pos, np.NPY_FLOAT64)
    rot_mat = check_array(rot_mat, np.NPY_FLOAT64)
    basis = check_array(basis, np.NPY_FLOAT64)

    cdef int i, j
    cdef double dx, dy, phi, theta, ax, ay, bx, by, cx, cy
    cdef int nf = fidxs.size - 1

    cdef np.uint32_t[::1] _x = x
    cdef np.uint32_t[::1] _y = y
    cdef np.int64_t[:, ::1] _hkl = hkl
    cdef np.uint32_t[::1] _fidxs = fidxs
    cdef np.float64_t[:, ::1] _smp_pos = smp_pos
    cdef np.float64_t[:, ::1] _rot_mat = rot_mat
    cdef np.float64_t[:, ::1] _basis = basis

    cdef np.npy_intp *shape = [x.shape[0], 2]
    cdef np.ndarray kin = np.PyArray_SimpleNew(2, shape, np.NPY_FLOAT32)
    cdef np.float32_t[:, ::1] _kin = kin

    for i in prange(nf, schedule='guided', num_threads=num_threads, nogil=True):
        for j in range(_fidxs[i], <int>_fidxs[i + 1]):
            dx = _x[j] * x_pixel_size - _smp_pos[i, 0]
            dy = _y[j] * y_pixel_size - _smp_pos[i, 1]
            phi = atan2(dy, dx)
            theta = atan(sqrt(dx * dx + dy * dy) / _smp_pos[i, 2])
            ax = _rot_mat[i, 0] * _basis[0, 0] + _rot_mat[i, 1] * _basis[0, 1] + _rot_mat[i, 2] * _basis[0, 2]
            ay = _rot_mat[i, 3] * _basis[0, 0] + _rot_mat[i, 4] * _basis[0, 1] + _rot_mat[i, 5] * _basis[0, 2]
            bx = _rot_mat[i, 0] * _basis[1, 0] + _rot_mat[i, 1] * _basis[1, 1] + _rot_mat[i, 2] * _basis[1, 2]
            by = _rot_mat[i, 3] * _basis[1, 0] + _rot_mat[i, 4] * _basis[1, 1] + _rot_mat[i, 5] * _basis[1, 2]
            cx = _rot_mat[i, 0] * _basis[2, 0] + _rot_mat[i, 1] * _basis[2, 1] + _rot_mat[i, 2] * _basis[2, 2]
            cy = _rot_mat[i, 3] * _basis[2, 0] + _rot_mat[i, 4] * _basis[2, 1] + _rot_mat[i, 5] * _basis[2, 2]
            _kin[j, 0] = sin(theta) * cos(phi) - ax * _hkl[j, 0] - bx * _hkl[j, 1] - cx * _hkl[j, 2]
            _kin[j, 1] = sin(theta) * sin(phi) - ay * _hkl[j, 0] - by * _hkl[j, 1] - cy * _hkl[j, 2]

    return kin

def update_sf(np.ndarray bp not None, np.ndarray sgn not None, np.ndarray xidx not None, np.ndarray xmap not None,
              np.ndarray xtal not None, np.ndarray hkl_idxs, np.ndarray iidxs, unsigned num_threads):
    if sgn.size != bp.size or sgn.size != xidx.size or sgn.size != xmap.shape[0]:
        raise ValueError('Input arrays have incompatible sizes')
    if sgn.size != iidxs[iidxs.size - 1]:
        raise ValueError('Input indices are incompatible with the input arrays')

    bp = check_array(bp, np.NPY_FLOAT32)
    sgn = check_array(sgn, np.NPY_FLOAT32)
    xidx = check_array(xidx, np.NPY_UINT32)
    xmap = check_array(xmap, np.NPY_FLOAT32)
    xtal = check_array(xtal, np.NPY_FLOAT32)
    hkl_idxs = check_array(hkl_idxs, np.NPY_UINT32)
    iidxs = check_array(iidxs, np.NPY_UINT32)

    cdef int fail = 0
    cdef unsigned long *_ddims = <unsigned long *>xtal.shape
    cdef unsigned long _hkl_size = np.PyArray_Max(hkl_idxs, 0, <np.ndarray>NULL) + 1
    cdef unsigned long _isize = iidxs.size - 1

    cdef np.ndarray sf = np.PyArray_SimpleNew(1, sgn.shape, np.NPY_FLOAT32)
    cdef float *_sf = <float *>np.PyArray_DATA(sf)
    cdef np.ndarray dsf = np.PyArray_SimpleNew(1, sgn.shape, np.NPY_FLOAT32)
    cdef float *_dsf = <float *>np.PyArray_DATA(dsf)
    cdef float *_bp = <float *>np.PyArray_DATA(bp)
    cdef float *_sgn = <float *>np.PyArray_DATA(sgn)
    cdef unsigned *_xidx = <unsigned *>np.PyArray_DATA(xidx)
    cdef float *_xmap = <float *>np.PyArray_DATA(xmap)
    cdef float *_xtal = <float *>np.PyArray_DATA(xtal)
    cdef unsigned *_hkl_idxs = <unsigned *>np.PyArray_DATA(hkl_idxs)
    cdef unsigned *_iidxs = <unsigned *>np.PyArray_DATA(iidxs)

    with nogil:
        fail = update_sf_c(_sf, _dsf, _bp, _sgn, _xidx, _xmap, _xtal, _ddims,
                           _hkl_idxs, _hkl_size, _iidxs, _isize, num_threads)

    if fail:
        raise RuntimeError('C backend exited with error.')

    return sf, dsf

def scaling_criterion(np.ndarray sf not None, np.ndarray bp not None, np.ndarray sgn not None, np.ndarray xidx not None,
                      np.ndarray xmap not None, np.ndarray xtal not None, np.ndarray iidxs, unsigned num_threads):
    if sf.size != bp.size or sf.size != sgn.size or sf.size != xidx.size or sf.size != xmap.shape[0]:
        raise ValueError('Input arrays have incompatible sizes')
    if sf.size != iidxs[iidxs.size - 1]:
        raise ValueError('Input indices are incompatible with the input arrays')

    sf = check_array(sf, np.NPY_FLOAT32)
    bp = check_array(bp, np.NPY_FLOAT32)
    sgn = check_array(sgn, np.NPY_FLOAT32)
    xidx = check_array(xidx, np.NPY_UINT32)
    xmap = check_array(xmap, np.NPY_FLOAT32)
    xtal = check_array(xtal, np.NPY_FLOAT32)
    iidxs = check_array(iidxs, np.NPY_UINT32)

    cdef np.npy_intp *new_dims
    cdef np.PyArray_Dims *new_shape
    if xtal.ndim == 2:
        new_dims = <np.npy_intp *>malloc(3 * sizeof(np.npy_intp))
        new_dims[0] = 1; new_dims[1] = xtal.shape[0]; new_dims[2] = xtal.shape[1]

        new_shape = <np.PyArray_Dims *>malloc(sizeof(np.PyArray_Dims))
        new_shape[0].ptr = new_dims; new_shape[0].len = 3

        xtal = np.PyArray_Newshape(xtal, new_shape, np.NPY_CORDER)
        free(new_dims); free(new_shape)

    if xtal.ndim != 3:
        raise ValueError('xtal has incompatible shape')

    cdef float err
    cdef unsigned long *_ddims = <unsigned long *>xtal.shape
    cdef unsigned long _isize = iidxs.size - 1

    cdef float *_sf = <float *>np.PyArray_DATA(sf)
    cdef float *_bp = <float *>np.PyArray_DATA(bp)
    cdef float *_sgn = <float *>np.PyArray_DATA(sgn)
    cdef unsigned *_xidx = <unsigned *>np.PyArray_DATA(xidx)
    cdef float *_xmap = <float *>np.PyArray_DATA(xmap)
    cdef float *_xtal = <float *>np.PyArray_DATA(xtal)
    cdef unsigned *_iidxs = <unsigned *>np.PyArray_DATA(iidxs)

    with nogil:
        err = scale_crit(_sf, _bp, _sgn, _xidx, _xmap, _xtal, _ddims, _iidxs, _isize, num_threads)

    return err

def kr_predict(np.ndarray y not None, np.ndarray x not None, np.ndarray x_hat, double sigma,
               double cutoff, np.ndarray w=None, unsigned int num_threads=1):
    y = check_array(y, np.NPY_FLOAT64)
    x = check_array(x, np.NPY_FLOAT64)
    x_hat = check_array(x_hat, np.NPY_FLOAT64)

    cdef int i
    cdef double *_w
    if w is None:
        w = <np.ndarray>np.PyArray_SimpleNew(y.ndim, y.shape, np.NPY_FLOAT64)
        _w = <double *>np.PyArray_DATA(w)
        for i in range(y.size):
            _w[i] = 1.0
    else:
        w = check_array(w, np.NPY_FLOAT64)
        _w = <double *>np.PyArray_DATA(w)

    cdef int fail = 0
    cdef np.npy_intp ndim = x.shape[x.ndim - 1]
    cdef np.npy_intp npts = x.size / ndim, nhat = x_hat.size / ndim

    if x.shape[x.ndim - 1] != x_hat.shape[x_hat.ndim - 1]:
        raise ValueError('`x` and `x_hat` have incompatible shapes')
    if npts != y.size:
        raise ValueError('`x` and `y` have incompatible shapes')

    cdef np.ndarray y_hat = <np.ndarray>np.PyArray_SimpleNew(x_hat.ndim - 1, x_hat.shape, np.NPY_FLOAT64)
    cdef double *_y_hat = <double *>np.PyArray_DATA(y_hat)
    cdef double *_y = <double *>np.PyArray_DATA(y)
    cdef double *_x = <double *>np.PyArray_DATA(x)
    cdef double *_x_hat = <double *>np.PyArray_DATA(x_hat)

    with nogil:
        fail = predict_kerreg(_y, _w, _x, npts, ndim, _y_hat, _x_hat, nhat, rbf,
                              sigma, cutoff, num_threads)

    if fail:
        raise RuntimeError('C backend exited with error.')

    return y_hat

def kr_grid(np.ndarray y not None, np.ndarray x not None, object step not None,
            double sigma, double cutoff, np.ndarray w=None, bint return_roi=True,
            unsigned int num_threads=1):
    y = check_array(y, np.NPY_FLOAT64)
    x = check_array(x, np.NPY_FLOAT64)

    cdef int i
    cdef double *_w
    if w is None:
        w = <np.ndarray>np.PyArray_SimpleNew(y.ndim, y.shape, np.NPY_FLOAT64)
        _w = <double *>np.PyArray_DATA(w)
        for i in range(y.size):
            _w[i] = 1.0
    else:
        w = check_array(w, np.NPY_FLOAT64)
        _w = <double *>np.PyArray_DATA(w)


    cdef int fail = 0
    cdef int npts = y.size
    cdef int ndim = x.shape[x.ndim - 1]

    cdef np.ndarray xstep = normalize_sequence(step, ndim, np.NPY_FLOAT64)
    cdef double *_step = <double *>np.PyArray_DATA(xstep)
    for i in range(ndim):
        if _step[i] <= 0.0:
            raise ValueError('step must be positive')

    cdef np.ndarray x_min = np.PyArray_Min(x, 0, <np.ndarray>NULL)
    cdef np.ndarray x_max = np.PyArray_Max(x, 0, <np.ndarray>NULL)
    cdef np.npy_intp *roi = <np.npy_intp *>malloc(2 * ndim * sizeof(np.npy_intp))
    cdef np.npy_intp *shape = <np.npy_intp *>malloc(ndim * sizeof(np.npy_intp))
    for i in range(ndim):
        roi[2 * i] = <int>floor(x_min[ndim - 1 - i] / step[ndim - 1 - i])
        roi[2 * i + 1] = <int>ceil((x_max[ndim - 1 - i] - x_min[ndim - 1 - i]) / step[ndim - 1 - i]) + 1
        shape[i] = roi[2 * i + 1] - roi[2 * i]

    cdef np.ndarray y_hat = <np.ndarray>np.PyArray_SimpleNew(ndim, shape, np.NPY_FLOAT64)
    cdef double *_y_hat = <double *>np.PyArray_DATA(y_hat)
    cdef double *_y = <double *>np.PyArray_DATA(y)
    cdef double *_x = <double *>np.PyArray_DATA(x)
    cdef unsigned long *_roi = <unsigned long *>roi

    with nogil:
        fail = predict_grid(_y, _w, _x, npts, ndim, _y_hat, _roi, _step, rbf,
                            sigma, cutoff, num_threads)

    if fail:
        raise RuntimeError('C backend exited with error.')

    cdef np.ndarray roi_arr

    if return_roi:
        roi_arr = ArrayWrapper.from_ptr(<void *>_roi).to_ndarray(1, [2 * ndim,], np.NPY_INTP)
        return y_hat, roi_arr
    else:
        free(roi)
        return y_hat

def xtal_interpolate(np.ndarray xidx not None, np.ndarray xmap not None,
                     np.ndarray xtal not None, unsigned num_threads=1):
    if xidx.size != xmap.shape[0]:
        raise ValueError('Input arrays have incompatible shapes')
    
    xidx = check_array(xidx, np.NPY_UINT32)
    xmap = check_array(xmap, np.NPY_FLOAT32)
    xtal = check_array(xtal, np.NPY_FLOAT32)

    cdef int fail = 0
    cdef unsigned long *_ddims = <unsigned long *>xtal.shape
    cdef unsigned long _isize = xidx.size

    cdef np.ndarray xtal_bi = np.PyArray_SimpleNew(1, xidx.shape, np.NPY_FLOAT32)
    cdef float *_xtal_bi = <float *>np.PyArray_DATA(xtal_bi)
    cdef unsigned *_xidx = <unsigned *>np.PyArray_DATA(xidx)
    cdef float *_xmap = <float *>np.PyArray_DATA(xmap)
    cdef float *_xtal = <float *>np.PyArray_DATA(xtal)

    with nogil:
        fail = xtal_interp(_xtal_bi, _xidx, _xmap, _xtal, _ddims, _isize, num_threads)

    if fail:
        raise RuntimeError('C backend exited with error.')

    return xtal_bi
