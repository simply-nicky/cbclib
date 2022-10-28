#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
import numpy as np
import cython
from libc.string cimport memcmp
from libc.math cimport ceil, exp, sqrt, atan, atan2, acos, fabs, sin, cos, log, floor
from libc.float cimport DBL_EPSILON
from cython.parallel import parallel, prange

# Numpy must be initialized. When using numpy from C or Cython you must
# *ALWAYS* do that, or you will have segfaults
np.import_array()

DEF N_BINS = 1024
DEF LINE_SIZE = 7

cdef line_profile profiles[3]
cdef void build_profiles():
    profiles[0] = linear_profile
    profiles[1] = quad_profile
    profiles[2] = tophat_profile

cdef dict profile_scheme
profile_scheme = {'linear': 0, 'quad': 1, 'tophat': 2}

build_profiles()

def draw_line_stack(np.ndarray inp not None, dict lines not None, int max_val=1, double dilation=0.0,
                     str profile='tophat', unsigned int num_threads=1):
    if inp.ndim < 2:
        raise ValueError('Input array must be >=2D array.')
    inp = check_array(inp, np.NPY_UINT32)

    if profile not in profile_scheme:
        raise ValueError(f"Invalid profile keyword: '{profile}'")

    cdef int ndim = inp.ndim
    cdef unsigned int *_inp = <unsigned int *>np.PyArray_DATA(inp)
    cdef int _X = <int>inp.shape[ndim - 1]
    cdef int _Y = <int>inp.shape[ndim - 2]
    cdef int repeats = inp.size / _X / _Y

    cdef int i, N = len(lines)
    cdef list frames = list(lines)
    cdef dict _lines = {}
    cdef float **_lptrs = <float **>malloc(N * sizeof(float *))
    cdef unsigned long **_ldims = <unsigned long **>malloc(N * sizeof(unsigned long *))
    for i in range(N):
        _lines[i] = check_array(lines[frames[i]], np.NPY_FLOAT32)
        _lptrs[i] = <float *>np.PyArray_DATA(_lines[i])
        _ldims[i] = <unsigned long *>(<np.ndarray>_lines[i]).shape

    cdef line_profile _prof = profiles[profile_scheme[profile]]

    if N < repeats:
        repeats = N
    num_threads = repeats if <int>num_threads > repeats else <int>num_threads
    cdef int fail

    for i in prange(repeats, schedule='guided', num_threads=num_threads, nogil=True):
        fail = draw_line_c(_inp + i * _Y * _X, _Y, _X, max_val, _lptrs[i], _ldims[i], <float>dilation, _prof)

        if fail:
            with gil:
                raise RuntimeError('C backend exited with error.')

    free(_lptrs); free(_ldims)

    return inp

def index_table(np.ndarray frames not None, np.ndarray indices not None):
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
        fail = index_table_c(&_funiq, &_fidxs, &fpts, &_iidxs, &ipts, _frames, _indices, npts)

    if fail:
        raise RuntimeError('C backend exited with error.')

    cdef np.npy_intp *shape = [fpts,]
    cdef np.ndarray funiq = ArrayWrapper.from_ptr(<void *>_funiq).to_ndarray(1, shape, np.NPY_UINT32)
    shape[0] = fpts + 1
    cdef np.ndarray fidxs = ArrayWrapper.from_ptr(<void *>_fidxs).to_ndarray(1, shape, np.NPY_UINT32)
    shape[0] = ipts
    cdef np.ndarray iidxs = ArrayWrapper.from_ptr(<void *>_iidxs).to_ndarray(1, shape, np.NPY_UINT32)
    return funiq, fidxs, iidxs

def calculate_kins(np.ndarray x not None, np.ndarray y not None, np.ndarray hkl not None, np.ndarray fidxs not None,
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

def update_sf(np.ndarray sgn not None, np.ndarray frames not None, np.ndarray kins not None,
              np.ndarray xtal not None, np.ndarray hkl_idxs, np.ndarray iidxs, unsigned num_threads):
    if sgn.size != frames.size or sgn.size != kins.shape[0]:
        raise ValueError('Input arrays have incompatible sizes')
    if sgn.size != iidxs[iidxs.size - 1]:
        raise ValueError('Input indices are incompatible with the input arrays')

    sgn = check_array(sgn, np.NPY_FLOAT32)
    frames = check_array(frames, np.NPY_UINT32)
    kins = check_array(kins, np.NPY_FLOAT32)
    xtal = check_array(xtal, np.NPY_FLOAT32)
    hkl_idxs = check_array(hkl_idxs, np.NPY_UINT32)
    iidxs = check_array(iidxs, np.NPY_UINT32)

    cdef int fail = 0
    cdef unsigned long *_ddims = <unsigned long *>xtal.shape
    cdef unsigned long _hkl_size = np.PyArray_Max(hkl_idxs, 0, <np.ndarray>NULL) + 1
    cdef unsigned long _isize = iidxs.size - 1

    cdef np.ndarray sf = np.PyArray_SimpleNew(1, sgn.shape, np.NPY_FLOAT32)
    cdef float *_sf = <float *>np.PyArray_DATA(sf)
    cdef float *_sgn = <float *>np.PyArray_DATA(sgn)
    cdef unsigned *_frames = <unsigned *>np.PyArray_DATA(frames)
    cdef float *_kins = <float *>np.PyArray_DATA(kins)
    cdef float *_xtal = <float *>np.PyArray_DATA(xtal)
    cdef unsigned *_hkl_idxs = <unsigned *>np.PyArray_DATA(hkl_idxs)
    cdef unsigned *_iidxs = <unsigned *>np.PyArray_DATA(iidxs)

    with nogil:
        fail = update_sf_c(_sf, _sgn, _frames, _kins, _xtal, _ddims,
                           _hkl_idxs, _hkl_size, _iidxs, _isize, num_threads)

    if fail:
        raise RuntimeError('C backend exited with error.')

    return sf

def scaling_criterion(np.ndarray sf not None, np.ndarray sgn not None, np.ndarray frames not None,
                      np.ndarray kins not None, np.ndarray xtal not None, np.ndarray iidxs, unsigned num_threads):
    if sf.size != sgn.size or sf.size != frames.size or sf.size != kins.shape[0]:
        raise ValueError('Input arrays have incompatible sizes')
    if sf.size != iidxs[iidxs.size - 1]:
        raise ValueError('Input indices are incompatible with the input arrays')

    sf = check_array(sf, np.NPY_FLOAT32)
    sgn = check_array(sgn, np.NPY_FLOAT32)
    frames = check_array(frames, np.NPY_UINT32)
    kins = check_array(kins, np.NPY_FLOAT32)
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
    cdef float *_sgn = <float *>np.PyArray_DATA(sgn)
    cdef unsigned *_frames = <unsigned *>np.PyArray_DATA(frames)
    cdef float *_kins = <float *>np.PyArray_DATA(kins)
    cdef float *_xtal = <float *>np.PyArray_DATA(xtal)
    cdef unsigned *_iidxs = <unsigned *>np.PyArray_DATA(iidxs)

    with nogil:
        err = scale_crit(_sf, _sgn, _frames, _kins, _xtal, _ddims, _iidxs, _isize, num_threads)

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
            double sigma, double cutoff, np.ndarray w=None, unsigned int num_threads=1):
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
    cdef np.npy_intp *shape = [<int>((x_max[0] - x_min[0]) / step[0]) + 1,
                               <int>((x_max[1] - x_min[1]) / step[1]) + 1]
    cdef unsigned long *_dims = <unsigned long *>shape

    cdef np.ndarray y_hat = <np.ndarray>np.PyArray_SimpleNew(ndim, shape, np.NPY_FLOAT64)
    cdef double *_y_hat = <double *>np.PyArray_DATA(y_hat)
    cdef double *_y = <double *>np.PyArray_DATA(y)
    cdef double *_x = <double *>np.PyArray_DATA(x)

    with nogil:
        fail = predict_grid(_y, _w, _x, npts, ndim, _y_hat, _dims, _step, rbf,
                            sigma, cutoff, num_threads)

    if fail:
        raise RuntimeError('C backend exited with error.')

    return y_hat
def searchsorted(np.ndarray a not None, np.ndarray v not None, str side='left'):
    if not np.PyArray_IS_C_CONTIGUOUS(a):
        a = np.PyArray_GETCONTIGUOUS(a)
    if not np.PyArray_IS_C_CONTIGUOUS(v):
        v = np.PyArray_GETCONTIGUOUS(v)

    cdef int type_num = np.PyArray_TYPE(a)
    if np.PyArray_TYPE(v) != type_num:
        raise ValueError('Input arrays have incompatible data types')
    
    cdef int i, i_end = v.size
    cdef unsigned long _asize = a.size
    cdef void *_a = np.PyArray_DATA(a)
    cdef void *_v = np.PyArray_DATA(v)
    cdef np.ndarray out = np.PyArray_SimpleNew(v.ndim, v.shape, np.NPY_UINT64)
    cdef unsigned long *_out = <unsigned long *>np.PyArray_DATA(out)
    cdef int _side = side_to_code(side)

    with nogil:
        if type_num == np.NPY_FLOAT64:
            for i in range(i_end):
                _out[i] = searchsorted_c(_v + i * sizeof(double), _a, _asize, sizeof(double), _side, compare_double)
        elif type_num == np.NPY_FLOAT32:
            for i in range(i_end):
                _out[i] = searchsorted_c(_v + i * sizeof(float), _a, _asize, sizeof(float), _side, compare_float)
        elif type_num == np.NPY_INT32:
            for i in range(i_end):
                _out[i] = searchsorted_c(_v + i * sizeof(int), _a, _asize, sizeof(int), _side, compare_int)
        elif type_num == np.NPY_UINT32:
            for i in range(i_end):
                _out[i] = searchsorted_c(_v + i * sizeof(unsigned int), _a, _asize, sizeof(unsigned int), _side, compare_uint)
        elif type_num == np.NPY_UINT64:
            for i in range(i_end):
                _out[i] = searchsorted_c(_v + i * sizeof(unsigned long), _a, _asize, sizeof(unsigned long), _side, compare_ulong)
        else:
            raise TypeError(f'a argument has incompatible type: {str(a.dtype)}')
    return out

def filter_hkl(np.ndarray sgn not None, np.ndarray bgd not None, np.ndarray coord not None,
               np.ndarray prob not None, np.ndarray idxs not None, double threshold,
               unsigned int num_threads=1):
    sgn = check_array(sgn, np.NPY_FLOAT32)
    bgd = check_array(bgd, np.NPY_FLOAT32)
    coord = check_array(coord, np.NPY_UINT32)
    prob = check_array(prob, np.NPY_FLOAT64)
    idxs = check_array(idxs, np.NPY_UINT64)

    cdef int i, i0, i1, n, n_max = np.PyArray_Max(idxs, 0, <np.ndarray>NULL)
    cdef unsigned long m, i_max = idxs.size
    cdef double I_sgn, I_bgd

    cdef np.float32_t[:, ::1] _sgn = sgn
    cdef np.float32_t[:, ::1] _bgd = bgd
    cdef np.uint32_t[:, ::1] _coord = coord
    cdef np.float64_t[::1] _prob = prob
    cdef void *_idxs = np.PyArray_DATA(idxs)

    cdef np.npy_intp *shape = [n_max + 1,]
    cdef np.ndarray out = np.PyArray_SimpleNew(1, shape, np.NPY_BOOL)
    cdef np.npy_bool[::1] _out = out

    for n in prange(n_max + 1, schedule='guided', num_threads=num_threads, nogil=True):
        m = n
        i0 = searchsorted_c(&m, _idxs, i_max, sizeof(unsigned long), SEARCH_LEFT, compare_ulong)
        m = n + 1
        i1 = searchsorted_c(&m, _idxs, i_max, sizeof(unsigned long), SEARCH_LEFT, compare_ulong)

        I_sgn = 0.0; I_bgd = 0.0
        for i in range(i0, i1):
            I_sgn = I_sgn + fabs(_sgn[_coord[i, 1], _coord[i, 0]]) * _prob[i]
            I_bgd = I_bgd + sqrt(_bgd[_coord[i, 1], _coord[i, 0]]) * _prob[i]
        _out[n] = I_sgn > threshold * I_bgd

    return out
