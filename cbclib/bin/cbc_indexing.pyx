cimport numpy as np
import numpy as np
import cython
from libc.stdlib cimport free, calloc, malloc, realloc
from libc.math cimport exp, sqrt, cos, sin, acos, atan2, fabs, log
from libc.float cimport DBL_EPSILON
from cython.parallel import parallel, prange
from .image_proc cimport check_array, normalize_sequence
from .line_detector cimport ArrayWrapper

# Numpy must be initialized. When using numpy from C or Cython you must
# *ALWAYS* do that, or you will have segfaults
np.import_array()

def euler_angles(np.ndarray rot_mats not None):
    rot_mats = check_array(rot_mats, np.NPY_FLOAT64)

    cdef np.npy_intp *new_dims
    cdef np.PyArray_Dims *new_shape
    if rot_mats.ndim == 2:
        new_dims = <np.npy_intp *>malloc(3 * sizeof(np.npy_intp))
        new_dims[0] = 1; new_dims[1] = rot_mats.shape[0]; new_dims[2] = rot_mats.shape[1]
   
        new_shape = <np.PyArray_Dims *>malloc(sizeof(np.PyArray_Dims))
        new_shape[0].ptr = new_dims; new_shape[0].len = 3

        rot_mats = np.PyArray_Newshape(rot_mats, new_shape, np.NPY_CORDER)
        free(new_dims); free(new_shape)    

    if rot_mats.ndim != 3 or (rot_mats.shape[1] != 3 or rot_mats.shape[2] != 3):
        raise ValueError('rot_mats has incompatible shape')

    cdef np.npy_intp *edims = [rot_mats.shape[0], 3]
    cdef np.ndarray angles = <np.ndarray>np.PyArray_SimpleNew(2, edims, np.NPY_FLOAT64)

    cdef double *e_ptr = <double *>np.PyArray_DATA(angles)
    cdef double *rm_ptr = <double *>np.PyArray_DATA(rot_mats)
    cdef unsigned long n_mats = rot_mats.shape[0]

    cdef int fail = 0
    with nogil:
        fail = compute_euler_angles(e_ptr, rm_ptr, n_mats)
    if fail:
        raise RuntimeError('C backend exited with error.')
    
    if rot_mats.shape[0] == 1:
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
    cdef np.ndarray rot_mats = <np.ndarray>np.PyArray_SimpleNew(3, rmdims, np.NPY_FLOAT64)

    cdef double *e_ptr = <double *>np.PyArray_DATA(angles)
    cdef double *rm_ptr = <double *>np.PyArray_DATA(rot_mats)
    cdef unsigned long n_mats = angles.shape[0]

    cdef int fail = 0
    with nogil:
        fail = compute_euler_matrix(rm_ptr, e_ptr, n_mats)
    if fail:
        raise RuntimeError('C backend exited with error.')

    if n_mats == 1:
        return rot_mats[0]
    return rot_mats

def tilt_angles(np.ndarray rot_mats not None):
    rot_mats = check_array(rot_mats, np.NPY_FLOAT64)

    cdef np.npy_intp *new_dims
    cdef np.PyArray_Dims *new_shape
    if rot_mats.ndim == 2:
        new_dims = <np.npy_intp *>malloc(3 * sizeof(np.npy_intp))
        new_dims[0] = 1; new_dims[1] = rot_mats.shape[0]; new_dims[2] = rot_mats.shape[1]
   
        new_shape = <np.PyArray_Dims *>malloc(sizeof(np.PyArray_Dims))
        new_shape[0].ptr = new_dims; new_shape[0].len = 3

        rot_mats = np.PyArray_Newshape(rot_mats, new_shape, np.NPY_CORDER)
        free(new_dims); free(new_shape)    

    if rot_mats.ndim != 3 or (rot_mats.shape[1] != 3 or rot_mats.shape[2] != 3):
        raise ValueError('rot_mats has incompatible shape')

    cdef np.npy_intp *edims = [rot_mats.shape[0], 3]
    cdef np.ndarray angles = <np.ndarray>np.PyArray_SimpleNew(2, edims, np.NPY_FLOAT64)

    cdef double *e_ptr = <double *>np.PyArray_DATA(angles)
    cdef double *rm_ptr = <double *>np.PyArray_DATA(rot_mats)
    cdef unsigned long n_mats = rot_mats.shape[0]

    cdef int fail = 0
    with nogil:
        fail = compute_tilt_angles(e_ptr, rm_ptr, n_mats)
    if fail:
        raise RuntimeError('C backend exited with error.')
    
    if rot_mats.shape[0] == 1:
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
    cdef np.ndarray rot_mats = <np.ndarray>np.PyArray_SimpleNew(3, rmdims, np.NPY_FLOAT64)

    cdef double *e_ptr = <double *>np.PyArray_DATA(angles)
    cdef double *rm_ptr = <double *>np.PyArray_DATA(rot_mats)
    cdef unsigned long n_mats = angles.shape[0]

    cdef int fail = 0
    with nogil:
        fail = compute_tilt_matrix(rm_ptr, e_ptr, n_mats)
    if fail:
        raise RuntimeError('C backend exited with error.')

    if n_mats == 1:
        return rot_mats[0]
    return rot_mats

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
    cdef np.ndarray rot_mats = <np.ndarray>np.PyArray_SimpleNew(3, rmdims, np.NPY_FLOAT64)

    cdef double *a_ptr = <double *>np.PyArray_DATA(a)
    cdef double *b_ptr = <double *>np.PyArray_DATA(b)
    cdef double *rm_ptr = <double *>np.PyArray_DATA(rot_mats)
    cdef unsigned long n_mats = a.shape[0]

    cdef int fail = 0
    with nogil:
        fail = compute_rotations(rm_ptr, a_ptr, b_ptr, n_mats)
    if fail:
        raise RuntimeError('C backend exited with error.')

    if n_mats == 1:
        return rot_mats[0]
    return rot_mats

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

def gaussian_grid(np.float64_t[:] x_arr, np.float64_t[:] y_arr, np.float64_t[:] z_arr,
                  np.float64_t[:, ::1] basis, double Sigma, double sigma, unsigned int threads=1):
    cdef int Nx = x_arr.size, Ny = y_arr.size, Nz = z_arr.size
    cdef double a_abs = basis[0, 0]**2 + basis[0, 1]**2 + basis[0, 2]**2
    cdef int na_min = <int>(((x_arr[0] if basis[0, 0] > 0.0 else x_arr[Nx - 1]) * basis[0, 0] + 
                             (y_arr[0] if basis[0, 1] > 0.0 else y_arr[Ny - 1]) * basis[0, 1] + 
                             (z_arr[0] if basis[0, 2] > 0.0 else z_arr[Nz - 1]) * basis[0, 2]) / a_abs)
    cdef int na_max = <int>(((x_arr[0] if basis[0, 0] <= 0.0 else x_arr[Nx - 1]) * basis[0, 0] + 
                             (y_arr[0] if basis[0, 1] <= 0.0 else y_arr[Ny - 1]) * basis[0, 1] + 
                             (z_arr[0] if basis[0, 2] <= 0.0 else z_arr[Nz - 1]) * basis[0, 2]) / a_abs)

    cdef double b_abs = basis[1, 0]**2 + basis[1, 1]**2 + basis[1, 2]**2
    cdef int nb_min = <int>(((x_arr[0] if basis[1, 0] > 0.0 else x_arr[Nx - 1]) * basis[1, 0] + 
                             (y_arr[0] if basis[1, 1] > 0.0 else y_arr[Ny - 1]) * basis[1, 1] + 
                             (z_arr[0] if basis[1, 2] > 0.0 else z_arr[Nz - 1]) * basis[1, 2]) / b_abs)
    cdef int nb_max = <int>(((x_arr[0] if basis[1, 0] <= 0.0 else x_arr[Nx - 1]) * basis[1, 0] + 
                             (y_arr[0] if basis[1, 1] <= 0.0 else y_arr[Ny - 1]) * basis[1, 1] + 
                             (z_arr[0] if basis[1, 2] <= 0.0 else z_arr[Nz - 1]) * basis[1, 2]) / b_abs)

    cdef double c_abs = basis[2, 0]**2 + basis[2, 1]**2 + basis[2, 2]**2
    cdef int nc_min = <int>(((x_arr[0] if basis[2, 0] > 0.0 else x_arr[Nx - 1]) * basis[2, 0] + 
                             (y_arr[0] if basis[2, 1] > 0.0 else y_arr[Ny - 1]) * basis[2, 1] + 
                             (z_arr[0] if basis[2, 2] > 0.0 else z_arr[Nz - 1]) * basis[2, 2]) / c_abs)
    cdef int nc_max = <int>(((x_arr[0] if basis[2, 0] <= 0.0 else x_arr[Nx - 1]) * basis[2, 0] + 
                             (y_arr[0] if basis[2, 1] <= 0.0 else y_arr[Ny - 1]) * basis[2, 1] + 
                             (z_arr[0] if basis[2, 2] <= 0.0 else z_arr[Nz - 1]) * basis[2, 2]) / c_abs)

    cdef np.npy_intp *odims = [Nx, Ny, Nz]
    cdef np.ndarray out = <np.ndarray>np.PyArray_ZEROS(3, odims, np.NPY_FLOAT64, 0)
    cdef np.float64_t[:, :, ::1] _out = out

    cdef int *hkl = <int *>malloc(3 * (na_max - na_min) * (nb_max - nb_min) * (nc_max - nc_min) * sizeof(int))
    cdef int na, nb, nc, n_max = 0
    cdef double cnt_x, cnt_y, cnt_z
    for na in range(na_min, na_max):
        for nb in range(nb_min, nb_max):
            for nc in range(nc_min, nc_max):
                cnt_x = na * basis[0, 0] + nb * basis[1, 0] + nc * basis[2, 0]
                cnt_y = na * basis[0, 1] + nb * basis[1, 1] + nc * basis[2, 1]
                cnt_z = na * basis[0, 2] + nb * basis[1, 2] + nc * basis[2, 2]
                if (cnt_x**2 + cnt_y**2 + cnt_z**2) < 9.0 * Sigma**2:
                    hkl[3 * n_max] = na; hkl[3 * n_max + 1] = nb; hkl[3 * n_max + 2] = nc; n_max += 1
                
    cdef int nx, ny, nz, n
    for nx in prange(Nx, schedule='guided', num_threads=threads, nogil=True):
        for ny in range(Ny):
            for nz in range(Nz):
                for n in range(n_max):
                    cnt_x = hkl[3 * n] * basis[0, 0] + hkl[3 * n + 1] * basis[1, 0] + hkl[3 * n + 2] * basis[2, 0]
                    cnt_y = hkl[3 * n] * basis[0, 1] + hkl[3 * n + 1] * basis[1, 1] + hkl[3 * n + 2] * basis[2, 1]
                    cnt_z = hkl[3 * n] * basis[0, 2] + hkl[3 * n + 1] * basis[1, 2] + hkl[3 * n + 2] * basis[2, 2]
                    _out[nx, ny, nz] = _out[nx, ny, nz] + exp(-0.5 * ((x_arr[nx] - cnt_x)**2 +
                                                                      (y_arr[ny] - cnt_y)**2 +
                                                                      (z_arr[nz] - cnt_z)**2) / (sigma * sigma))
                _out[nx, ny, nz] = _out[nx, ny, nz] * exp(-0.5 * (x_arr[nx]**2 + y_arr[ny]**2 + z_arr[nz]**2) / (Sigma * Sigma))

    hkl = <int *>realloc(hkl, 3 * n_max * sizeof(int))
    cdef np.npy_intp *hkl_dims = [n_max, 3]
    cdef np.ndarray hkl_arr = ArrayWrapper.from_ptr(<void *>hkl).to_ndarray(2, hkl_dims, np.NPY_INT32)

    return out, hkl_arr

def gaussian_grid_grad(np.float64_t[:] x_arr, np.float64_t[:] y_arr, np.float64_t[:] z_arr,
                       np.float64_t[:, ::1] basis, np.int32_t[:, ::1] hkl, double Sigma, double sigma,
                       unsigned int threads=1):
    cdef int Nx = x_arr.size, Ny = y_arr.size, Nz = z_arr.size
    cdef np.npy_intp *odims = [9, Nx, Ny, Nz]
    cdef np.ndarray out = <np.ndarray>np.PyArray_ZEROS(4, odims, np.NPY_FLOAT64, 0)
    cdef np.float64_t[:, :, :, ::1] _out = out
                
    cdef int nx, ny, nz, n, n_max = hkl.shape[0], i
    cdef double gauss, dr
    cdef double *cnt
    with nogil, parallel(num_threads=threads):
        cnt = <double *>malloc(3 * sizeof(double))

        for nx in prange(Nx, schedule='guided'):
            for ny in range(Ny):
                for nz in range(Nz):
                    for n in range(n_max):
                        cnt[0] = hkl[n, 0] * basis[0, 0] + hkl[n, 1] * basis[1, 0] + hkl[n, 2] * basis[2, 0]
                        cnt[1] = hkl[n, 0] * basis[0, 1] + hkl[n, 1] * basis[1, 1] + hkl[n, 2] * basis[2, 1]
                        cnt[2] = hkl[n, 0] * basis[0, 2] + hkl[n, 1] * basis[1, 2] + hkl[n, 2] * basis[2, 2]
                        dr = (x_arr[nx] - cnt[0])**2 + (y_arr[ny] - cnt[1])**2 + (z_arr[nz] - cnt[2])**2
                        gauss = exp(-0.5 * dr / (sigma * sigma))

                        _out[0, nx, ny, nz] = _out[0, nx, ny, nz] + (x_arr[nx] - cnt[0]) * hkl[n, 0] * gauss / (sigma * sigma)
                        _out[1, nx, ny, nz] = _out[1, nx, ny, nz] + (y_arr[ny] - cnt[1]) * hkl[n, 0] * gauss / (sigma * sigma)
                        _out[2, nx, ny, nz] = _out[2, nx, ny, nz] + (z_arr[nz] - cnt[2]) * hkl[n, 0] * gauss / (sigma * sigma)
                        _out[3, nx, ny, nz] = _out[3, nx, ny, nz] + (x_arr[nx] - cnt[0]) * hkl[n, 1] * gauss / (sigma * sigma)
                        _out[4, nx, ny, nz] = _out[4, nx, ny, nz] + (y_arr[ny] - cnt[1]) * hkl[n, 1] * gauss / (sigma * sigma)
                        _out[5, nx, ny, nz] = _out[5, nx, ny, nz] + (z_arr[nz] - cnt[2]) * hkl[n, 1] * gauss / (sigma * sigma)
                        _out[6, nx, ny, nz] = _out[6, nx, ny, nz] + (x_arr[nx] - cnt[0]) * hkl[n, 2] * gauss / (sigma * sigma)
                        _out[7, nx, ny, nz] = _out[7, nx, ny, nz] + (y_arr[ny] - cnt[1]) * hkl[n, 2] * gauss / (sigma * sigma)
                        _out[8, nx, ny, nz] = _out[8, nx, ny, nz] + (z_arr[nz] - cnt[2]) * hkl[n, 2] * gauss / (sigma * sigma)
                    
                    gauss = exp(-0.5 * (x_arr[nx]**2 + y_arr[ny]**2 + z_arr[nz]**2) / (Sigma * Sigma))
                    for i in range(9):
                        _out[i, nx, ny, nz] = _out[i, nx, ny, nz] * gauss

        free(cnt)

    return out

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

def calc_source_lines(np.float64_t[:, ::1] basis, np.int64_t[:, ::1] hkl, np.float64_t[::1] kin_min,
                      np.float64_t[::1] kin_max, unsigned int threads=1):
    cdef int n_max = hkl.shape[0], n, i, j
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

    cdef np.npy_intp *odims = [n_max, 2, 3]
    cdef np.ndarray out = <np.ndarray>np.PyArray_ZEROS(3, odims, np.NPY_FLOAT64, 0)
    cdef unsigned char *_mask = <unsigned char *>calloc(n_max, sizeof(unsigned char))
    cdef np.float64_t[:, :, ::1] _out = out

    cdef double src_th, src_prd, t_int
    cdef double *q
    cdef double *q_sph

    with nogil, parallel(num_threads=threads):
        q = <double *>malloc(3 * sizeof(double))
        q_sph = <double *>malloc(3 * sizeof(double))    # r, theta, phi
        t_int = 0.0

        for n in prange(n_max, schedule='guided'):
            q[0] = hkl[n, 0] * basis[0, 0] + hkl[n, 1] * basis[1, 0] + hkl[n, 2] * basis[2, 0]
            q[1] = hkl[n, 0] * basis[0, 1] + hkl[n, 1] * basis[1, 1] + hkl[n, 2] * basis[2, 1]
            q[2] = hkl[n, 0] * basis[0, 2] + hkl[n, 1] * basis[1, 2] + hkl[n, 2] * basis[2, 2]
            q_sph[0] = sqrt(q[0]**2 + q[1]**2 + q[2]**2)
            q_sph[1] = acos(-q[2] / q_sph[0])
            q_sph[2] = atan2(q[1], q[0])

            src_th = q_sph[1] - acos(0.5 * q_sph[0])
            if fabs(sin(src_th)) < sqrt(kin_max[0]**2 + kin_max[1]**2):
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

def cross_entropy(np.int64_t[::1] x, np.float64_t[::1] p, np.uint32_t[::1] q, int q_max, double epsilon): 
    cdef double entropy = 0.0
    cdef int i, n = x.size
    with nogil:
        for i in range(n):
            entropy -= p[i] * log(<double>(q[x[i]]) / q_max + epsilon)
    return entropy / n
