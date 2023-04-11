#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
import numpy as np
import cython
from libc.string cimport memcmp, memcpy
from libc.math cimport ceil, exp, sqrt, atan, atan2, acos, fabs, sin, cos, log, floor
from libc.float cimport DBL_EPSILON
from cython.parallel import parallel, prange

cdef line_profile profiles[4]
cdef void build_profiles():
    profiles[0] = linear_profile
    profiles[1] = quad_profile
    profiles[2] = tophat_profile
    profiles[3] = gauss_profile

cdef dict profile_scheme
profile_scheme = {'linear': 0, 'quad': 1, 'tophat': 2, 'gauss': 3}

build_profiles()

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

DEF N_BINS = 1024
DEF LINE_SIZE = 7

def det_to_k(np.ndarray x not None, np.ndarray y not None, np.ndarray src not None, np.ndarray idxs=None,
             unsigned num_threads=1):
    cdef unsigned long _ksize = x.size

    if src.shape[src.ndim - 1] != 3:
        raise ValueError('src has invalid shape')
    cdef unsigned long _ssize = src.size / src.shape[src.ndim - 1]

    if _ssize == 1:
        idxs = np.PyArray_ZEROS(1, [_ksize,], np.NPY_UINT32, 0)
    if idxs is None:
        raise ValueError('idxs is not provided')
    if idxs[idxs.size - 1] + 1 > _ssize:
        raise ValueError('src is incompatible with idxs')
    
    if _ksize != y.size or _ksize != idxs.size:
        raise ValueError('x, y, and idxs have incompatible shapes')

    x = check_array(x, np.NPY_FLOAT64)
    y = check_array(y, np.NPY_FLOAT64)
    idxs = check_array(idxs, np.NPY_UINT32)
    src = check_array(src, np.NPY_FLOAT64)

    cdef int i
    cdef np.npy_intp *kdims = <np.npy_intp *>malloc((x.ndim + 1) * sizeof(np.npy_intp))
    for i in range(x.ndim):
        kdims[i] = x.shape[i]
    kdims[x.ndim] = 3
    cdef np.ndarray karr = np.PyArray_SimpleNew(x.ndim + 1, kdims, np.NPY_FLOAT64)

    free(kdims)

    cdef double *_karr = <double *>np.PyArray_DATA(karr)
    cdef double *_x = <double *>np.PyArray_DATA(x)
    cdef double *_y = <double *>np.PyArray_DATA(y)
    cdef unsigned *_idxs = <unsigned *>np.PyArray_DATA(idxs)
    cdef double *_src = <double *>np.PyArray_DATA(src)
    cdef int fail = 0

    with nogil:
        fail = det2k(_karr, _x, _y, _idxs, _ksize, _src, num_threads)
    
    if fail:
        raise RuntimeError('C backend exited with error.')

    return karr

def k_to_det(np.ndarray karr not None, np.ndarray src not None, np.ndarray idxs=None, unsigned num_threads=1):
    if karr.shape[karr.ndim - 1] != 3:
        raise ValueError('karr has an invalid shape')
    cdef unsigned long _ksize = karr.size / karr.shape[karr.ndim - 1]

    if src.shape[src.ndim - 1] != 3:
        raise ValueError('src has invalid shape')
    cdef unsigned long _ssize = src.size / src.shape[src.ndim - 1]

    if _ssize == 1:
        idxs = np.PyArray_ZEROS(1, [_ksize,], np.NPY_UINT32, 0)
    if idxs is None:
        raise ValueError('idxs is not provided')
    if idxs[idxs.size - 1] + 1 > _ssize:
        raise ValueError('src is incompatible with idxs')

    if _ksize != idxs.size:
        raise ValueError('karr and idxs have incompatible shapes')

    karr = check_array(karr, np.NPY_FLOAT64)
    idxs = check_array(idxs, np.NPY_UINT32)
    src = check_array(src, np.NPY_FLOAT64)

    cdef np.ndarray x = np.PyArray_SimpleNew(karr.ndim - 1, karr.shape, np.NPY_FLOAT64)
    cdef np.ndarray y = np.PyArray_SimpleNew(karr.ndim - 1, karr.shape, np.NPY_FLOAT64)

    cdef double *_karr = <double *>np.PyArray_DATA(karr)
    cdef double *_x = <double *>np.PyArray_DATA(x)
    cdef double *_y = <double *>np.PyArray_DATA(y)
    cdef unsigned *_idxs = <unsigned *>np.PyArray_DATA(idxs)
    cdef double *_src = <double *>np.PyArray_DATA(src)

    with nogil:
        fail = k2det(_x, _y, _karr, _idxs, _ksize, _src, num_threads)
    
    if fail:
        raise RuntimeError('C backend exited with error.')

    return x, y

def rotate(np.ndarray vecs not None, np.ndarray rmats not None, np.ndarray idxs=None, unsigned num_threads=1):
    if vecs.shape[vecs.ndim - 1] != 3:
        raise ValueError('vecs has an invalid shape')
    cdef unsigned long _vsize = vecs.size / vecs.shape[vecs.ndim - 1]

    if rmats.shape[rmats.ndim - 1] != 3 or rmats.shape[rmats.ndim - 2] != 3:
        raise ValueError('rmats has invalid shape')
    cdef unsigned long _rsize = rmats.size / (rmats.shape[rmats.ndim - 1] * rmats.shape[rmats.ndim - 2])

    if _rsize == 1:
        idxs = np.PyArray_ZEROS(1, [_vsize,], np.NPY_UINT32, 0)
    if idxs is None:
        raise ValueError('idxs is not provided')
    if idxs[idxs.size - 1] + 1 > _rsize:
        raise ValueError('rmats is incompatible with idxs')

    if _vsize != idxs.size:
        raise ValueError('vecs and idxs have incompatible shapes')  

    vecs = check_array(vecs, np.NPY_FLOAT64)
    idxs = check_array(idxs, np.NPY_UINT32)
    rmats = check_array(rmats, np.NPY_FLOAT64)

    cdef np.ndarray out = np.PyArray_SimpleNew(vecs.ndim, vecs.shape, np.NPY_FLOAT64)

    cdef double *_out = <double *>np.PyArray_DATA(out)
    cdef double *_vecs = <double *>np.PyArray_DATA(vecs)
    cdef unsigned *_idxs = <unsigned *>np.PyArray_DATA(idxs)
    cdef double *_rmats = <double *>np.PyArray_DATA(rmats)
    cdef int fail = 0

    with nogil:
        fail = rotate_vec(_out, _vecs, _idxs, _vsize, _rmats, num_threads)

    if fail:
        raise RuntimeError('C backend exited with error.')

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
