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

# Numpy must be initialized. When using numpy from C or Cython you must
# *ALWAYS* do that, or you will have segfaults
np.import_array()

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

    x = check_array(x, np.NPY_FLOAT32)
    y = check_array(y, np.NPY_FLOAT32)
    idxs = check_array(idxs, np.NPY_UINT32)
    src = check_array(src, np.NPY_FLOAT32)

    cdef int i
    cdef np.npy_intp *kdims = <np.npy_intp *>malloc((x.ndim + 1) * sizeof(np.npy_intp))
    for i in range(x.ndim):
        kdims[i] = x.shape[i]
    kdims[x.ndim] = 3
    cdef np.ndarray karr = np.PyArray_SimpleNew(x.ndim + 1, kdims, np.NPY_FLOAT32)

    free(kdims)

    cdef float *_karr = <float *>np.PyArray_DATA(karr)
    cdef float *_x = <float *>np.PyArray_DATA(x)
    cdef float *_y = <float *>np.PyArray_DATA(y)
    cdef unsigned *_idxs = <unsigned *>np.PyArray_DATA(idxs)
    cdef float *_src = <float *>np.PyArray_DATA(src)

    cdef int fail = 0
    with nogil:
        fail = det2k(_karr, _x, _y, _idxs, _ksize, _src, num_threads)
    if fail:
        raise RuntimeError('C backend exited with error.')

    return karr

def det_to_k_vjp(np.ndarray vec not None, np.ndarray x not None, np.ndarray y not None, np.ndarray src not None,
                 np.ndarray idxs=None, unsigned num_threads=1):
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
    
    if _ksize != y.size or _ksize != idxs.size or <int>_ksize != vec.shape[0]:
        raise ValueError('x, y, vec, and idxs have incompatible shapes')

    vec = check_array(vec, np.NPY_FLOAT32)
    x = check_array(x, np.NPY_FLOAT32)
    y = check_array(y, np.NPY_FLOAT32)
    idxs = check_array(idxs, np.NPY_UINT32)
    src = check_array(src, np.NPY_FLOAT32)

    cdef np.ndarray xout = np.PyArray_SimpleNew(x.ndim, x.shape, np.NPY_FLOAT32)
    cdef np.ndarray yout = np.PyArray_SimpleNew(y.ndim, y.shape, np.NPY_FLOAT32)
    cdef np.ndarray sout = np.PyArray_ZEROS(src.ndim, src.shape, np.NPY_FLOAT32, 0)

    cdef float *_xout = <float *>np.PyArray_DATA(xout)
    cdef float *_yout = <float *>np.PyArray_DATA(yout)
    cdef float *_sout = <float *>np.PyArray_DATA(sout)
    cdef float *_vec = <float *>np.PyArray_DATA(vec)
    cdef float *_x = <float *>np.PyArray_DATA(x)
    cdef float *_y = <float *>np.PyArray_DATA(y)
    cdef unsigned *_idxs = <unsigned *>np.PyArray_DATA(idxs)
    cdef float *_src = <float *>np.PyArray_DATA(src)

    cdef int fail = 0
    with nogil:
        fail = det2k_vjp(_xout, _yout, _sout, _vec, _x, _y, _idxs, _ksize, _src,
                         _ssize, num_threads)
    if fail:
        raise RuntimeError('C backend exited with error.')

    return xout, yout, sout

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

    karr = check_array(karr, np.NPY_FLOAT32)
    idxs = check_array(idxs, np.NPY_UINT32)
    src = check_array(src, np.NPY_FLOAT32)

    cdef np.ndarray x = np.PyArray_SimpleNew(karr.ndim - 1, karr.shape, np.NPY_FLOAT32)
    cdef np.ndarray y = np.PyArray_SimpleNew(karr.ndim - 1, karr.shape, np.NPY_FLOAT32)

    cdef float *_karr = <float *>np.PyArray_DATA(karr)
    cdef float *_x = <float *>np.PyArray_DATA(x)
    cdef float *_y = <float *>np.PyArray_DATA(y)
    cdef unsigned *_idxs = <unsigned *>np.PyArray_DATA(idxs)
    cdef float *_src = <float *>np.PyArray_DATA(src)

    cdef int fail = 0
    with nogil:
        fail = k2det(_x, _y, _karr, _idxs, _ksize, _src, num_threads)
    if fail:
        raise RuntimeError('C backend exited with error.')

    return x, y

def k_to_det_vjp(np.ndarray xvec not None, np.ndarray yvec not None, np.ndarray karr not None, np.ndarray src not None,
                 np.ndarray idxs=None, unsigned num_threads=1):
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

    if _ksize != idxs.size or _ksize != xvec.size or _ksize != yvec.size:
        raise ValueError('xvec, yvec, karr, and idxs have incompatible shapes')

    xvec = check_array(xvec, np.NPY_FLOAT32)
    yvec = check_array(yvec, np.NPY_FLOAT32)
    karr = check_array(karr, np.NPY_FLOAT32)
    idxs = check_array(idxs, np.NPY_UINT32)
    src = check_array(src, np.NPY_FLOAT32)

    cdef np.ndarray kout = np.PyArray_SimpleNew(karr.ndim, karr.shape, np.NPY_FLOAT32)
    cdef np.ndarray sout = np.PyArray_ZEROS(src.ndim, src.shape, np.NPY_FLOAT32, 0)

    cdef float *_kout = <float *>np.PyArray_DATA(kout)
    cdef float *_sout = <float *>np.PyArray_DATA(sout)
    cdef float *_xvec = <float *>np.PyArray_DATA(xvec)
    cdef float *_yvec = <float *>np.PyArray_DATA(yvec)
    cdef float *_karr = <float *>np.PyArray_DATA(karr)
    cdef unsigned *_idxs = <unsigned *>np.PyArray_DATA(idxs)
    cdef float *_src = <float *>np.PyArray_DATA(src)

    cdef int fail = 0
    with nogil:
        fail = k2det_vjp(_kout, _sout, _xvec, _yvec, _karr, _idxs, _ksize, _src,
                         _ssize, num_threads)
    if fail:
        raise RuntimeError('C backend exited with error.')

    return kout, sout

def k_to_smp(np.ndarray karr not None, np.ndarray z not None, np.ndarray src not None, np.ndarray idxs=None,
             unsigned num_threads=1):
    if karr.shape[karr.ndim - 1] != 3:
        raise ValueError('karr has an invalid shape')
    cdef unsigned long _ksize = karr.size / karr.shape[karr.ndim - 1]

    if src.ndim != 1 or src.size != 3:
        raise ValueError('src has invalid shape')

    if z.size == 1:
        idxs = np.PyArray_ZEROS(1, [_ksize,], np.NPY_UINT32, 0)
    if idxs is None:
        raise ValueError('idxs is not provided')
    if idxs[idxs.size - 1] + 1 > z.size:
        raise ValueError('src is incompatible with idxs')

    if _ksize != idxs.size:
        raise ValueError('karr and idxs have incompatible shapes')

    karr = check_array(karr, np.NPY_FLOAT32)
    idxs = check_array(idxs, np.NPY_UINT32)
    z = check_array(z, np.NPY_FLOAT32)
    src = check_array(src, np.NPY_FLOAT32)

    cdef np.ndarray pts = np.PyArray_SimpleNew(karr.ndim, karr.shape, np.NPY_FLOAT32)

    cdef float *_pts = <float *>np.PyArray_DATA(pts)
    cdef float *_karr = <float *>np.PyArray_DATA(karr)
    cdef unsigned *_idxs = <unsigned *>np.PyArray_DATA(idxs)
    cdef float *_z = <float *>np.PyArray_DATA(z)
    cdef float *_src = <float *>np.PyArray_DATA(src)

    cdef int fail = 0
    with nogil:
        fail = k2smp(_pts, _karr, _idxs, _ksize, _z, _src, num_threads)
    if fail:
        raise RuntimeError('C backend exited with error.')

    return pts

def k_to_smp_vjp(np.ndarray vec not None, np.ndarray karr not None, np.ndarray z not None, np.ndarray src not None,
                 np.ndarray idxs=None, unsigned num_threads=1):
    if karr.shape[karr.ndim - 1] != 3:
        raise ValueError('karr has an invalid shape')
    cdef unsigned long _ksize = karr.size / karr.shape[karr.ndim - 1]

    if src.ndim != 1 or src.size != 3:
        raise ValueError('src has invalid shape')

    if z.size == 1:
        idxs = np.PyArray_ZEROS(1, [_ksize,], np.NPY_UINT32, 0)
    if idxs is None:
        raise ValueError('idxs is not provided')
    if idxs[idxs.size - 1] + 1 > z.size:
        raise ValueError('src is incompatible with idxs')

    if _ksize != idxs.size or <int>_ksize != vec.shape[0]:
        raise ValueError('vec, karr, and idxs have incompatible shapes')

    vec = check_array(vec, np.NPY_FLOAT32)
    karr = check_array(karr, np.NPY_FLOAT32)
    idxs = check_array(idxs, np.NPY_UINT32)
    z = check_array(z, np.NPY_FLOAT32)
    src = check_array(src, np.NPY_FLOAT32)

    cdef np.ndarray kout = np.PyArray_SimpleNew(karr.ndim, karr.shape, np.NPY_FLOAT32)
    cdef np.ndarray zout = np.PyArray_ZEROS(z.ndim, z.shape, np.NPY_FLOAT32, 0)
    cdef np.ndarray sout = np.PyArray_ZEROS(src.ndim, src.shape, np.NPY_FLOAT32, 0)

    cdef float *_kout = <float *>np.PyArray_DATA(kout)
    cdef float *_zout = <float *>np.PyArray_DATA(zout)
    cdef float *_sout = <float *>np.PyArray_DATA(sout)
    cdef float *_vec = <float *>np.PyArray_DATA(vec)
    cdef float *_karr = <float *>np.PyArray_DATA(karr)
    cdef unsigned *_idxs = <unsigned *>np.PyArray_DATA(idxs)
    cdef float *_z = <float *>np.PyArray_DATA(z)
    cdef unsigned long _zsize = z.size
    cdef float *_src = <float *>np.PyArray_DATA(src)

    cdef int fail = 0
    with nogil:
        fail = k2smp_vjp(_kout, _zout, _sout, _vec, _karr, _idxs, _ksize, _z, _zsize, _src, num_threads)
    if fail:
        raise RuntimeError('C backend exited with error.')

    return kout, zout, sout

def source_lines(np.ndarray hkl not None, np.ndarray basis not None, np.ndarray kmin not None,
                 np.ndarray kmax not None, np.ndarray hidxs=None, np.ndarray bidxs=None,
                 unsigned num_threads=1):
    if hkl.shape[hkl.ndim - 1] != 3:
        raise ValueError('hkl has incompatible shape')
    cdef unsigned long _hsize = hkl.size / hkl.shape[hkl.ndim - 1]

    if basis.shape[basis.ndim - 1] != 9:
        raise ValueError('basis has incompatible shape')
    cdef unsigned long _bsize = basis.size / basis.shape[basis.ndim - 1]

    cdef int i
    cdef unsigned long _N
    if hidxs is None and bidxs is None:
        _N = _hsize * _bsize
        hidxs = np.PyArray_SimpleNew(1, [_N,], np.NPY_UINT32)
        bidxs = np.PyArray_SimpleNew(1, [_N,], np.NPY_UINT32)
        for i in range(<int>_N):
            hidxs[i] = i % _hsize
            bidxs[i] = i // _hsize
    elif hidxs is None and _hsize == 1:
        _N = _bsize
        hidxs = np.PyArray_ZEROS(1, [_N,], np.NPY_UINT32, 0)
    elif bidxs is None and _bsize == 1:
        _N = _hsize
        bidxs = np.PyArray_ZEROS(1, [_N,], np.NPY_UINT32, 0)
    else:
        if hidxs.size != bidxs.size:
            raise ValueError('hidxs and bidxs have incompatible shapes')

        _N = hidxs.size

    hkl = check_array(hkl, np.NPY_INT32)
    basis = check_array(basis, np.NPY_FLOAT32)
    hidxs = check_array(hidxs, np.NPY_UINT32)
    bidxs = check_array(bidxs, np.NPY_UINT32)

    cdef np.ndarray out = np.PyArray_SimpleNew(3, [_N, 2, 3], np.NPY_FLOAT32)
    cdef np.ndarray mask = np.PyArray_SimpleNew(1, [_N], np.NPY_BOOL)

    cdef float *_out = <float *>np.PyArray_DATA(out)
    cdef unsigned char *_mask = <unsigned char *>np.PyArray_DATA(mask)
    cdef int *_hkl = <int *>np.PyArray_DATA(hkl)
    cdef unsigned *_hidxs = <unsigned *>np.PyArray_DATA(hidxs)
    cdef float *_basis = <float *>np.PyArray_DATA(basis)
    cdef unsigned *_bidxs = <unsigned *>np.PyArray_DATA(bidxs)
    cdef float[4] _pupil = [kmin[0], kmin[1], kmax[0], kmax[1]]

    cdef int fail = 0
    with nogil:
        fail = find_kins(_out, _mask, _N, _hkl, _hidxs, _basis, _bidxs, _pupil, num_threads)
    if fail:
        raise RuntimeError('C backend exited with error.')

    return out, mask

def source_lines_vjp(np.ndarray vec not None, np.ndarray hkl not None, np.ndarray basis not None,
                     np.ndarray kmin not None, np.ndarray kmax not None, np.ndarray hidxs=None,
                     np.ndarray bidxs=None, unsigned num_threads=1):
    if hkl.shape[hkl.ndim - 1] != 3:
        raise ValueError('hkl has incompatible shape')
    cdef unsigned long _hsize = hkl.size / hkl.shape[hkl.ndim - 1]

    if basis.shape[basis.ndim - 1] != 9:
        raise ValueError('basis has incompatible shape')
    cdef unsigned long _bsize = basis.size / basis.shape[basis.ndim - 1]

    cdef int i
    cdef unsigned long _N
    if hidxs is None and bidxs is None:
        _N = _hsize * _bsize
        hidxs = np.PyArray_SimpleNew(1, [_N,], np.NPY_UINT32)
        bidxs = np.PyArray_SimpleNew(1, [_N,], np.NPY_UINT32)
        for i in range(<int>_N):
            hidxs[i] = i % _hsize
            bidxs[i] = i // _hsize
    elif hidxs is None and _hsize == 1:
        _N = _bsize
        hidxs = np.PyArray_ZEROS(1, [_N,], np.NPY_UINT32, 0)
    elif bidxs is None and _bsize == 1:
        _N = _hsize
        bidxs = np.PyArray_ZEROS(1, [_N,], np.NPY_UINT32, 0)
    else:
        if hidxs.size != bidxs.size:
            raise ValueError('hidxs and bidxs have incompatible shapes')

        _N = hidxs.size

    if <int>_N != vec.shape[0] or vec.shape[1] != 2 or vec.shape[2] != 3:
        raise ValueError('vec has incompatible shape')

    hkl = check_array(hkl, np.NPY_INT32)
    basis = check_array(basis, np.NPY_FLOAT32)
    hidxs = check_array(hidxs, np.NPY_UINT32)
    bidxs = check_array(bidxs, np.NPY_UINT32)

    cdef np.ndarray bout = np.PyArray_ZEROS(basis.ndim, basis.shape, np.NPY_FLOAT32, 0)
    cdef np.ndarray kout = np.PyArray_ZEROS(1, [4,], np.NPY_FLOAT32, 0)

    cdef float *_bout = <float *>np.PyArray_DATA(bout)
    cdef float *_kout = <float *>np.PyArray_DATA(kout)
    cdef float *_vec = <float *>np.PyArray_DATA(vec)
    cdef int *_hkl = <int *>np.PyArray_DATA(hkl)
    cdef unsigned *_hidxs = <unsigned *>np.PyArray_DATA(hidxs)
    cdef float *_basis = <float *>np.PyArray_DATA(basis)
    cdef unsigned *_bidxs = <unsigned *>np.PyArray_DATA(bidxs)
    cdef float[4] _pupil = [kmin[0], kmin[1], kmax[0], kmax[1]]

    cdef int fail = 0
    with nogil:
        fail = find_kins_vjp(_bout, _kout, _vec, _N, _hkl, _hidxs, _basis, _bsize,
                             _bidxs, _pupil, num_threads)
    if fail:
        raise RuntimeError('C backend exited with error.')

    return bout, kout