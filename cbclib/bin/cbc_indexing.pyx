import numpy as np
from cython.parallel import parallel, prange
from .image_proc cimport check_array, normalize_sequence
from .line_detector cimport ArrayWrapper

# Numpy must be initialized. When using numpy from C or Cython you must
# *ALWAYS* do that, or you will have segfaults
np.import_array()

DEF CUTOFF = 3.0

def euler_angles(np.ndarray rmats not None):
    rmats = check_array(rmats, np.NPY_FLOAT64)

    cdef np.npy_intp *new_dims
    cdef np.PyArray_Dims *new_shape
    if rmats.ndim == 2:
        new_dims = <np.npy_intp *>malloc(3 * sizeof(np.npy_intp))
        new_dims[0] = 1; new_dims[1] = rmats.shape[0]; new_dims[2] = rmats.shape[1]
   
        new_shape = <np.PyArray_Dims *>malloc(sizeof(np.PyArray_Dims))
        new_shape[0].ptr = new_dims; new_shape[0].len = 3

        rmats = np.PyArray_Newshape(rmats, new_shape, np.NPY_CORDER)
        free(new_dims); free(new_shape)    

    if rmats.ndim != 3 or (rmats.shape[1] != 3 or rmats.shape[2] != 3):
        raise ValueError('rmats has incompatible shape')

    cdef np.npy_intp *edims = [rmats.shape[0], 3]
    cdef np.ndarray angles = <np.ndarray>np.PyArray_SimpleNew(2, edims, np.NPY_FLOAT64)

    cdef double *e_ptr = <double *>np.PyArray_DATA(angles)
    cdef double *rm_ptr = <double *>np.PyArray_DATA(rmats)
    cdef unsigned long n_mats = rmats.shape[0]

    cdef int fail = 0
    with nogil:
        fail = compute_euler_angles(e_ptr, rm_ptr, n_mats)
    if fail:
        raise RuntimeError('C backend exited with error.')
    
    if rmats.shape[0] == 1:
        return angles[0]
    return angles

def euler_matrix(np.ndarray angles not None):
    angles = check_array(angles, np.NPY_FLOAT64)

    cdef np.npy_intp *new_dims
    cdef np.PyArray_Dims *new_shape
    if angles.ndim == 1:
        new_dims = <np.npy_intp *>malloc(2 * sizeof(np.npy_intp))
        new_dims[0] = 1; new_dims[1] = angles.shape[0]

        new_shape = <np.PyArray_Dims *>malloc(sizeof(np.PyArray_Dims))
        new_shape[0].ptr = new_dims; new_shape[0].len = 2
        
        angles = np.PyArray_Newshape(angles, new_shape, np.NPY_CORDER)
        free(new_dims); free(new_shape)

    if angles.ndim != 2 or angles.shape[1] != 3:
        raise ValueError('angles has incompatible shape')

    cdef np.npy_intp *rmdims = [angles.shape[0], 3, 3]
    cdef np.ndarray rmats = <np.ndarray>np.PyArray_SimpleNew(3, rmdims, np.NPY_FLOAT64)

    cdef double *e_ptr = <double *>np.PyArray_DATA(angles)
    cdef double *rm_ptr = <double *>np.PyArray_DATA(rmats)
    cdef unsigned long n_mats = angles.shape[0]

    cdef int fail = 0
    with nogil:
        fail = compute_euler_matrix(rm_ptr, e_ptr, n_mats)
    if fail:
        raise RuntimeError('C backend exited with error.')

    if n_mats == 1:
        return rmats[0]
    return rmats

def tilt_angles(np.ndarray rmats not None):
    rmats = check_array(rmats, np.NPY_FLOAT64)

    cdef np.npy_intp *new_dims
    cdef np.PyArray_Dims *new_shape
    if rmats.ndim == 2:
        new_dims = <np.npy_intp *>malloc(3 * sizeof(np.npy_intp))
        new_dims[0] = 1; new_dims[1] = rmats.shape[0]; new_dims[2] = rmats.shape[1]
   
        new_shape = <np.PyArray_Dims *>malloc(sizeof(np.PyArray_Dims))
        new_shape[0].ptr = new_dims; new_shape[0].len = 3

        rmats = np.PyArray_Newshape(rmats, new_shape, np.NPY_CORDER)
        free(new_dims); free(new_shape)    

    if rmats.ndim != 3 or (rmats.shape[1] != 3 or rmats.shape[2] != 3):
        raise ValueError('rmats has incompatible shape')

    cdef np.npy_intp *edims = [rmats.shape[0], 3]
    cdef np.ndarray angles = <np.ndarray>np.PyArray_SimpleNew(2, edims, np.NPY_FLOAT64)

    cdef double *e_ptr = <double *>np.PyArray_DATA(angles)
    cdef double *rm_ptr = <double *>np.PyArray_DATA(rmats)
    cdef unsigned long n_mats = rmats.shape[0]

    cdef int fail = 0
    with nogil:
        fail = compute_tilt_angles(e_ptr, rm_ptr, n_mats)
    if fail:
        raise RuntimeError('C backend exited with error.')
    
    if rmats.shape[0] == 1:
        return angles[0]
    return angles

def tilt_matrix(np.ndarray angles not None):
    angles = check_array(angles, np.NPY_FLOAT64)

    cdef np.npy_intp *new_dims
    cdef np.PyArray_Dims *new_shape
    if angles.ndim == 1:
        new_dims = <np.npy_intp *>malloc(2 * sizeof(np.npy_intp))
        new_dims[0] = 1; new_dims[1] = angles.shape[0]

        new_shape = <np.PyArray_Dims *>malloc(sizeof(np.PyArray_Dims))
        new_shape[0].ptr = new_dims; new_shape[0].len = 2
        
        angles = np.PyArray_Newshape(angles, new_shape, np.NPY_CORDER)
        free(new_dims); free(new_shape)

    if angles.ndim != 2 or angles.shape[1] != 3:
        raise ValueError('angles has incompatible shape')

    cdef np.npy_intp *rmdims = [angles.shape[0], 3, 3]
    cdef np.ndarray rmats = <np.ndarray>np.PyArray_SimpleNew(3, rmdims, np.NPY_FLOAT64)

    cdef double *e_ptr = <double *>np.PyArray_DATA(angles)
    cdef double *rm_ptr = <double *>np.PyArray_DATA(rmats)
    cdef unsigned long n_mats = angles.shape[0]

    cdef int fail = 0
    with nogil:
        fail = compute_tilt_matrix(rm_ptr, e_ptr, n_mats)
    if fail:
        raise RuntimeError('C backend exited with error.')

    if n_mats == 1:
        return rmats[0]
    return rmats

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

    karr = check_array(karr, np.NPY_FLOAT64)
    idxs = check_array(idxs, np.NPY_UINT32)
    z = check_array(z, np.NPY_FLOAT64)
    src = check_array(src, np.NPY_FLOAT64)

    cdef np.ndarray pts = np.PyArray_SimpleNew(karr.ndim, karr.shape, np.NPY_FLOAT64)

    cdef double *_pts = <double *>np.PyArray_DATA(pts)
    cdef double *_karr = <double *>np.PyArray_DATA(karr)
    cdef unsigned *_idxs = <unsigned *>np.PyArray_DATA(idxs)
    cdef double *_z = <double *>np.PyArray_DATA(z)
    cdef double *_src = <double *>np.PyArray_DATA(src)

    with nogil:
        fail = k2smp(_pts, _karr, _idxs, _ksize, _z, _src, num_threads)
    
    if fail:
        raise RuntimeError('C backend exited with error.')

    return pts

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

def find_rotations(np.ndarray a not None, np.ndarray b not None):
    a = check_array(a, np.NPY_FLOAT64)

    cdef np.npy_intp *new_dims
    cdef np.PyArray_Dims *new_shape
    if a.ndim == 1:
        new_dims = <np.npy_intp *>malloc(2 * sizeof(np.npy_intp))
        new_dims[0] = 1; new_dims[1] = a.shape[0]

        new_shape = <np.PyArray_Dims *>malloc(sizeof(np.PyArray_Dims))
        new_shape[0].ptr = new_dims; new_shape[0].len = 2
        
        a = np.PyArray_Newshape(a, new_shape, np.NPY_CORDER)
        free(new_dims); free(new_shape)

    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError('a has incompatible shape')

    b = check_array(b, np.NPY_FLOAT64)

    if b.ndim == 1:
        new_dims = <np.npy_intp *>malloc(2 * sizeof(np.npy_intp))
        new_dims[0] = 1; new_dims[1] = b.shape[0]

        new_shape = <np.PyArray_Dims *>malloc(sizeof(np.PyArray_Dims))
        new_shape[0].ptr = new_dims; new_shape[0].len = 2
        
        b = np.PyArray_Newshape(b, new_shape, np.NPY_CORDER)
        free(new_dims); free(new_shape)

    if b.ndim != 2 or b.shape[1] != 3 or b.shape[0] != a.shape[0]:
        raise ValueError('b has incompatible shape')

    cdef np.npy_intp *rmdims = [a.shape[0], 3, 3]
    cdef np.ndarray rmats = <np.ndarray>np.PyArray_SimpleNew(3, rmdims, np.NPY_FLOAT64)

    cdef double *a_ptr = <double *>np.PyArray_DATA(a)
    cdef double *b_ptr = <double *>np.PyArray_DATA(b)
    cdef double *rm_ptr = <double *>np.PyArray_DATA(rmats)
    cdef unsigned long n_mats = a.shape[0]

    cdef int fail = 0
    with nogil:
        fail = compute_rotations(rm_ptr, a_ptr, b_ptr, n_mats)
    if fail:
        raise RuntimeError('C backend exited with error.')

    if n_mats == 1:
        return rmats[0]
    return rmats

def cartesian_to_spherical(np.ndarray vecs not None):
    vecs = check_array(vecs, np.NPY_FLOAT64)

    cdef np.npy_intp *new_dims
    cdef np.PyArray_Dims *new_shape
    if vecs.ndim == 1:
        new_dims = <np.npy_intp *>malloc(2 * sizeof(np.npy_intp))
        new_dims[0] = 1; new_dims[1] = vecs.shape[0]

        new_shape = <np.PyArray_Dims *>malloc(sizeof(np.PyArray_Dims))
        new_shape[0].ptr = new_dims; new_shape[0].len = 2
        
        vecs = np.PyArray_Newshape(vecs, new_shape, np.NPY_CORDER)
        free(new_dims); free(new_shape)

    if vecs.ndim != 2 or vecs.shape[1] != 3:
        raise ValueError('vecs has incompatible shape')

    cdef np.npy_intp *odims = [vecs.shape[0], 3]
    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(2, odims, np.NPY_FLOAT64)
    cdef np.float64_t[:, ::1] _out = out
    cdef np.float64_t[:, ::1] _vecs = vecs
    cdef int n_vecs = vecs.shape[0], i

    with nogil:
        for i in range(n_vecs):
            _out[i, 0] = sqrt(_vecs[i, 0]**2 + _vecs[i, 1]**2 + _vecs[i, 2]**2)
            _out[i, 1] = acos(_vecs[i, 2] / _out[i, 0])
            _out[i, 2] = atan2(_vecs[i, 1], _vecs[i, 0])
    
    if n_vecs == 1:
        return out[0]
    return out

def spherical_to_cartesian(np.ndarray vecs not None):
    vecs = check_array(vecs, np.NPY_FLOAT64)

    cdef np.npy_intp *new_dims
    cdef np.PyArray_Dims *new_shape
    if vecs.ndim == 1:
        new_dims = <np.npy_intp *>malloc(2 * sizeof(np.npy_intp))
        new_dims[0] = 1; new_dims[1] = vecs.shape[0]

        new_shape = <np.PyArray_Dims *>malloc(sizeof(np.PyArray_Dims))
        new_shape[0].ptr = new_dims; new_shape[0].len = 2
        
        vecs = np.PyArray_Newshape(vecs, new_shape, np.NPY_CORDER)
        free(new_dims); free(new_shape)

    if vecs.ndim != 2 or vecs.shape[1] != 3:
        raise ValueError('vecs has incompatible shape')

    cdef np.npy_intp *odims = [vecs.shape[0], 3]
    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(2, odims, np.NPY_FLOAT64)
    cdef np.float64_t[:, ::1] _out = out
    cdef np.float64_t[:, ::1] _vecs = vecs
    cdef int n_vecs = vecs.shape[0], i

    with nogil:
        for i in range(n_vecs):
            _out[i, 0] = _vecs[i, 0] * sin(_vecs[i, 1]) * cos(_vecs[i, 2])
            _out[i, 1] = _vecs[i, 0] * sin(_vecs[i, 1]) * sin(_vecs[i, 2])
            _out[i, 2] = _vecs[i, 0] * cos(_vecs[i, 1])
    
    if n_vecs == 1:
        return out[0]
    return out

def filter_direction(np.ndarray grid not None, object axis not None, double rng, double sigma, int num_threads=1):
    grid = check_array(grid, np.NPY_FLOAT64)

    cdef np.ndarray ax = normalize_sequence(axis, 3, np.NPY_FLOAT64)
    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(grid.ndim - 1, grid.shape, np.NPY_FLOAT64)

    cdef np.float64_t[:, :, :, ::1] _grid = grid
    cdef np.float64_t[:, :, ::1] _out = out
    cdef np.float64_t[::1] _axis = ax
    cdef double norm = _axis[0] * _axis[0] + _axis[1] * _axis[1] + _axis[2] * _axis[2]
    cdef double rng_sq = rng * rng
    cdef double sgm_sq = sigma * sigma
    cdef int i, j, k
    cdef double prod, dx, dy, dz, val0, val1

    for i in prange(_out.shape[0], schedule='guided', num_threads=num_threads, nogil=True):
        for j in range(_out.shape[1]):
            for k in range(_out.shape[2]):
                prod = _grid[i, j, k, 0] * _axis[0] + _grid[i, j, k, 1] * _axis[1] + _grid[i, j, k, 2] * _axis[2]
                dx = _grid[i, j, k, 0] - prod * _axis[0] / norm
                dy = _grid[i, j, k, 1] - prod * _axis[1] / norm
                dz = _grid[i, j, k, 2] - prod * _axis[2] / norm
                val0 = 1.0 / (1.0 + exp((rng_sq - dx * dx - dy * dy - dz * dz) / sgm_sq))
                val1 = 1.0 / (1.0 + exp((_grid[i, j, k, 0] * _grid[i, j, k, 0] +
                                         _grid[i, j, k, 1] * _grid[i, j, k, 1] +
                                         _grid[i, j, k, 2] * _grid[i, j, k, 2] - rng_sq) / sgm_sq))
                _out[i, j, k] = val0 if val0 > val1 else val1
    return out

def gaussian_grid(np.float64_t[:, :, ::1] p_arr, np.float64_t[::1] x_arr, np.float64_t[::1] y_arr,
                  np.float64_t[::1] z_arr, np.float64_t[::1] center, np.float64_t[:, ::1] basis, double sigma,
                  double cutoff, unsigned int num_threads=1):
    cdef int Nx = x_arr.size, Ny = y_arr.size, Nz = z_arr.size, i
    cdef int n_min[3]
    cdef int n_max[3]
    cdef double vec_abs
    for i in range(3):
        vec_abs = basis[i, 0]**2 + basis[i, 1]**2 + basis[i, 2]**2
        if vec_abs:
            n_min[i] = <int>(((x_arr[0] if basis[i, 0] > 0.0 else x_arr[Nx - 1]) * basis[i, 0] + 
                              (y_arr[0] if basis[i, 1] > 0.0 else y_arr[Ny - 1]) * basis[i, 1] + 
                              (z_arr[0] if basis[i, 2] > 0.0 else z_arr[Nz - 1]) * basis[i, 2]) / vec_abs)
            n_max[i] = <int>(((x_arr[0] if basis[i, 0] <= 0.0 else x_arr[Nx - 1]) * basis[i, 0] + 
                              (y_arr[0] if basis[i, 1] <= 0.0 else y_arr[Ny - 1]) * basis[i, 1] + 
                              (z_arr[0] if basis[i, 2] <= 0.0 else z_arr[Nz - 1]) * basis[i, 2]) / vec_abs)
        else:
            n_min[i] = 0
            n_max[i] = 1

    cdef int *hkl = <int *>malloc(3 * (n_max[0] - n_min[0]) *
                                      (n_max[1] - n_min[1]) *
                                      (n_max[2] - n_min[2]) * sizeof(int))
    cdef double *qs = <double *>malloc(3 * (n_max[0] - n_min[0]) *
                                           (n_max[1] - n_min[1]) *
                                           (n_max[2] - n_min[2]) * sizeof(double))
    cdef int na, nb, nc, hkl_max = 0
    for na in range(n_min[0], n_max[0]):
        for nb in range(n_min[1], n_max[1]):
            for nc in range(n_min[2], n_max[2]):
                qs[3 * hkl_max] = na * basis[0, 0] + nb * basis[1, 0] + nc * basis[2, 0] + center[0]
                qs[3 * hkl_max + 1] = na * basis[0, 1] + nb * basis[1, 1] + nc * basis[2, 1] + center[1]
                qs[3 * hkl_max + 2] = na * basis[0, 2] + nb * basis[1, 2] + nc * basis[2, 2] + center[2]
                if (qs[3 * hkl_max]**2 + qs[3 * hkl_max + 1]**2 + qs[3 * hkl_max + 2]**2) < cutoff**2:
                    hkl[3 * hkl_max] = na; hkl[3 * hkl_max + 1] = nb; hkl[3 * hkl_max + 2] = nc; hkl_max += 1
                
    cdef int nx, ny, nz, t
    cdef double q, x_min, x_max, y_min, y_max, z_min, z_max, entropy = 0.0
    cdef double grad_xx = 0.0, grad_xy = 0.0, grad_xz = 0.0
    cdef double grad_yx = 0.0, grad_yy = 0.0, grad_yz = 0.0
    cdef double grad_zx = 0.0, grad_zy = 0.0, grad_zz = 0.0
    cdef np.ndarray grad = np.PyArray_SimpleNew(1, [9,], np.NPY_FLOAT64)
    cdef int *min_buf
    cdef int *max_buf
    with nogil, parallel(num_threads=num_threads):
        min_buf = <int *>malloc(3 * sizeof(int))
        max_buf = <int *>malloc(3 * sizeof(int))
        t = openmp.omp_get_thread_num()

        for i in prange(hkl_max):
            x_min = qs[3 * i] - CUTOFF * sigma
            x_max = qs[3 * i] + CUTOFF * sigma
            y_min = qs[3 * i + 1] - CUTOFF * sigma
            y_max = qs[3 * i + 1] + CUTOFF * sigma
            z_min = qs[3 * i + 2] - CUTOFF * sigma
            z_max = qs[3 * i + 2] + CUTOFF * sigma

            min_buf[0] = searchsorted(&x_min, &x_arr[0], Nx, sizeof(double), SEARCH_LEFT, compare_double)
            max_buf[0] = searchsorted(&x_max, &x_arr[0], Nx, sizeof(double), SEARCH_LEFT, compare_double)
            min_buf[1] = searchsorted(&y_min, &y_arr[0], Ny, sizeof(double), SEARCH_LEFT, compare_double)
            max_buf[1] = searchsorted(&y_max, &y_arr[0], Ny, sizeof(double), SEARCH_LEFT, compare_double)
            min_buf[2] = searchsorted(&z_min, &z_arr[0], Nz, sizeof(double), SEARCH_LEFT, compare_double)
            max_buf[2] = searchsorted(&z_max, &z_arr[0], Nz, sizeof(double), SEARCH_LEFT, compare_double)

            for nx in range(min_buf[0], max_buf[0] + 1 if max_buf[0] < Nx else Nx):
                for ny in range(min_buf[1], max_buf[1] + 1 if max_buf[1] < Ny else Ny):
                    for nz in range(min_buf[2], max_buf[2] + 1 if max_buf[2] < Nz else Nz):
                        q = exp(-0.5 * ((x_arr[nx] - qs[3 * i])**2 +
                                        (y_arr[ny] - qs[3 * i + 1])**2 +
                                        (z_arr[nz] - qs[3 * i + 2])**2) / (sigma * sigma))
                        entropy -= p_arr[nz, ny, nx] * q
                        grad_xx -= p_arr[nz, ny, nx] * (x_arr[nx] - qs[3 * i]) * hkl[3 * i] * q / (sigma * sigma)
                        grad_xy -= p_arr[nz, ny, nx] * (y_arr[ny] - qs[3 * i + 1]) * hkl[3 * i] * q / (sigma * sigma)
                        grad_xz -= p_arr[nz, ny, nx] * (z_arr[nz] - qs[3 * i + 2]) * hkl[3 * i] * q / (sigma * sigma)
                        grad_yx -= p_arr[nz, ny, nx] * (x_arr[nx] - qs[3 * i]) * hkl[3 * i + 1] * q / (sigma * sigma)
                        grad_yy -= p_arr[nz, ny, nx] * (y_arr[ny] - qs[3 * i + 1]) * hkl[3 * i + 1] * q / (sigma * sigma)
                        grad_yz -= p_arr[nz, ny, nx] * (z_arr[nz] - qs[3 * i + 2]) * hkl[3 * i + 1] * q / (sigma * sigma)
                        grad_zx -= p_arr[nz, ny, nx] * (x_arr[nx] - qs[3 * i]) * hkl[3 * i + 2] * q / (sigma * sigma)
                        grad_zy -= p_arr[nz, ny, nx] * (y_arr[ny] - qs[3 * i + 1]) * hkl[3 * i + 2] * q / (sigma * sigma)
                        grad_zz -= p_arr[nz, ny, nx] * (z_arr[nz] - qs[3 * i + 2]) * hkl[3 * i + 2] * q / (sigma * sigma)

        free(min_buf); free(max_buf)

    grad[0] = grad_xx; grad[1] = grad_xy; grad[2] = grad_xz
    grad[3] = grad_yx; grad[4] = grad_yy; grad[5] = grad_yz
    grad[6] = grad_zx; grad[7] = grad_zy; grad[8] = grad_zz

    free(qs); free(hkl)

    return entropy, grad

cdef int find_intersection(double *t_int, double *q, double *e, double *s, double *tlim, double src_prd) nogil:
    # ----------------------------------
    #  f1 = o . q - s . q    f2 = q . e
    # ----------------------------------
    cdef double f1 = src_prd - s[0] * q[0] - s[1] * q[1]
    cdef double f2 = e[0] * q[0] + e[1] * q[1]

    # ----------------------------------
    #  Solving a quadratic equation:
    #  a * t^2 - 2 b * t + c = 0
    #  a = f2^2 + q_z^2     b = f1 * f2
    #  c = f1^2 - (1 - s^2) * q_z^2
    # ----------------------------------
    cdef double a = f2 * f2 + q[2] * q[2]
    cdef double b = f1 * f2
    cdef double c = f1 * f1 - (1.0 - s[0] * s[0] - s[1] * s[1]) * q[2] * q[2]

    cdef double delta = sqrt(b * b - a * c), t, x, y, prd

    if delta >= 0.0:
        t = (b - delta) / a
        x = s[0] + t * e[0]
        y = s[1] + t * e[1]
        prd = x * q[0] + y * q[1] + sqrt(1.0 - x * x - y * y) * q[2] - src_prd
        if t >= tlim[0] and t <= tlim[1] and fabs(prd) < DBL_EPSILON:
            t_int[0] = t
            return 1

        t = (b + delta) / a
        x = s[0] + t * e[0]
        y = s[1] + t * e[1]
        prd = x * q[0] + y * q[1] + sqrt(1.0 - x * x - y * y) * q[2] - src_prd
        if t >= tlim[0] and t <= tlim[1] and fabs(prd) < DBL_EPSILON:
            t_int[0] = t
            return 1

        return 0
    
    return 0

def calc_source_lines(np.ndarray basis not None, np.ndarray hkl not None, np.ndarray kin_min not None,
                      np.ndarray kin_max not None, int num_threads=1):
    basis = check_array(basis, np.NPY_FLOAT64)
    hkl = check_array(hkl, np.NPY_INT64)

    cdef np.float64_t[:, ::1] _basis = basis
    cdef np.int64_t[:, ::1] _hkl = hkl

    cdef int n_max = _hkl.shape[0], n, i, j
    cdef double[4][2] bs = [[kin_min[0], 0.0], [0.0, kin_min[1]], [0.0, kin_max[1]], [kin_max[0], 0.0]]
    cdef double[4][2] taus = [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    cdef double[4][2] tlim = [[taus[0][0] * kin_min[0] + taus[0][1] * kin_min[1],
                               taus[0][0] * kin_max[0] + taus[0][1] * kin_max[1]],
                              [taus[1][0] * kin_min[0] + taus[1][1] * kin_min[1],
                               taus[1][0] * kin_max[0] + taus[1][1] * kin_max[1]],
                              [taus[2][0] * kin_min[0] + taus[2][1] * kin_min[1],
                               taus[2][0] * kin_max[0] + taus[2][1] * kin_max[1]],
                              [taus[3][0] * kin_min[0] + taus[3][1] * kin_min[1],
                               taus[3][0] * kin_max[0] + taus[3][1] * kin_max[1]],]
    cdef double NA = sqrt(kin_max[0]**2 + kin_max[1]**2)

    cdef np.npy_intp *odims = [n_max, 2, 3]
    cdef np.ndarray out = <np.ndarray>np.PyArray_ZEROS(3, odims, np.NPY_FLOAT64, 0)
    cdef unsigned char *_mask = <unsigned char *>calloc(n_max, sizeof(unsigned char))
    cdef np.float64_t[:, :, ::1] _out = out

    cdef double src_th, src_prd, t_int
    cdef double *q
    cdef double *q_sph

    with nogil, parallel(num_threads=num_threads):
        q = <double *>malloc(3 * sizeof(double))
        q_sph = <double *>malloc(3 * sizeof(double))    # r, theta, phi
        t_int = 0.0

        for n in prange(n_max, schedule='guided'):
            q[0] = _hkl[n, 0] * _basis[0, 0] + _hkl[n, 1] * _basis[1, 0] + _hkl[n, 2] * _basis[2, 0]
            q[1] = _hkl[n, 0] * _basis[0, 1] + _hkl[n, 1] * _basis[1, 1] + _hkl[n, 2] * _basis[2, 1]
            q[2] = _hkl[n, 0] * _basis[0, 2] + _hkl[n, 1] * _basis[1, 2] + _hkl[n, 2] * _basis[2, 2]
            q_sph[0] = sqrt(q[0]**2 + q[1]**2 + q[2]**2)
            q_sph[1] = acos(-q[2] / q_sph[0])
            q_sph[2] = atan2(q[1], q[0])

            src_th = q_sph[1] - acos(0.5 * q_sph[0])
            if fabs(sin(src_th)) < NA:
                src_prd = -sin(src_th) * cos(q_sph[2]) * q[0] - sin(src_th) * sin(q_sph[2]) * q[1] + cos(src_th) * q[2]

                i = 0; j = 0
                while j < 2 and i < 4:
                    if find_intersection(&t_int, q, taus[i], bs[i], tlim[i], src_prd):
                        _out[n, j, 0] = bs[i][0] + t_int * taus[i][0]
                        _out[n, j, 1] = bs[i][1] + t_int * taus[i][1]
                        _out[n, j, 2] = sqrt(1.0 - _out[n, j, 0]**2 - _out[n, j, 1]**2)
                        j = j + 1
                    i = i + 1

                if j == 2:
                    _mask[n] = 1
        
        free(q)
        free(q_sph)

    cdef np.ndarray mask = ArrayWrapper.from_ptr(<void *>_mask).to_ndarray(1, odims, np.NPY_BOOL)
    out = np.PyArray_Compress(out, mask, 0, <np.ndarray>NULL)
    return out, mask
