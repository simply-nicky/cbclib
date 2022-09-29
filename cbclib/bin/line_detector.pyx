import numpy as np
from cython.parallel import prange, parallel

# Numpy must be initialized. When using numpy from C or Cython you must
# *ALWAYS* do that, or you will have segfaults
np.import_array()

DEF N_BINS = 1024
DEF LINE_SIZE = 7

cdef class ArrayWrapper:
    """A wrapper class for a C data structure. """

    def __cinit__(self):
        self._data = NULL

    def __dealloc__(self):
        if not self._data is NULL:
            free(self._data)
            self._data = NULL

cdef class LSD:

    def __cinit__(self, float scale=0.8, float sigma_scale=0.6, float log_eps=0.0,
                  float ang_th=45.0, float density_th=0.7, float quant=2.0):
        if scale < 0 or scale > 1:
            raise ValueError('scale is out of bounds (0.0, 1.0)')
        else:
            self.scale = scale
        if sigma_scale < 0 or sigma_scale > 1:
            raise ValueError('sigma_scale is out of bounds (0.0, 1.0)')
        else:
            self.sigma_scale = sigma_scale
        self.log_eps = log_eps
        if ang_th < 0 or ang_th > 360:
            raise ValueError('ang_th is out of bounds (0.0, 360.0)')
        else:
            self.ang_th = ang_th
        if density_th < 0 or density_th > 1:
            raise ValueError('density_th is out of bounds (0.0, 1.0)')
        else:
            self.density_th = density_th
        if quant < 0:
            raise ValueError('quant msut be positive')
        else:
            self.quant = quant

    def detect(self, np.ndarray image not None, float cutoff, float filter_threshold=0.0,
               float group_threshold=0.6, int n_group=2, bint filter=True, bint group=True, 
               float dilation=0.0, bint return_labels=False, unsigned int num_threads=1):
        if image.ndim < 2:
            raise ValueError('Image must be a 2D array.')
        image = check_array(image, np.NPY_FLOAT32)

        cdef int ndim = image.ndim
        cdef float *_img = <float *>np.PyArray_DATA(image)
        cdef int _X = <int>image.shape[ndim - 1]
        cdef int _Y = <int>image.shape[ndim - 2]
        cdef int repeats = image.size / _X / _Y
        cdef np.ndarray streaks, cond

        cdef float **_outs = <float **>malloc(repeats * sizeof(float *))
        if _outs is NULL:
            raise MemoryError('not enough memory')

        cdef unsigned char **_masks = <unsigned char **>malloc(repeats * sizeof(unsigned char *))
        if _masks is NULL:
            raise MemoryError('not enough memory')

        cdef int *_ns = <int *>malloc(repeats * sizeof(int))
        if _ns is NULL:
            free(_outs)
            raise MemoryError('not enough memory')

        cdef int **_regs = <int **>malloc(repeats * sizeof(int *))
        if _regs is NULL:
            free(_outs); free(_ns)
            raise MemoryError('not enough memory')

        cdef int *_reg_xs = <int *>malloc(repeats * sizeof(int))
        if _reg_xs is NULL:
            free(_outs); free(_ns); free(_regs)
            raise MemoryError('not enough memory')
        
        cdef int *_reg_ys = <int *>malloc(repeats * sizeof(int))
        if _reg_ys is NULL:
            free(_outs); free(_ns); free(_regs); free(_reg_xs)
            raise MemoryError('not enough memory')

        cdef int fail = 0, i, j
        cdef dict line_dict = {}, reg_dict = {}, out_dict = {}
        cdef np.npy_intp *out_dims = [0, LINE_SIZE]
        cdef unsigned long *ldims

        num_threads = repeats if <int>num_threads > repeats else <int>num_threads

        with nogil, parallel(num_threads=num_threads):
            ldims = <unsigned long *>malloc(2 * sizeof(unsigned long))
            ldims[1] = LINE_SIZE

            for i in prange(repeats, schedule='guided'):
                fail |= LineSegmentDetection(&_outs[i], &_ns[i], _img + i * _Y * _X, _Y, _X,
                                             self.scale, self.sigma_scale, self.quant,
                                             self.ang_th, self.log_eps, self.density_th, N_BINS,
                                             &_regs[i], &_reg_ys[i], &_reg_xs[i])

                _masks[i] = <unsigned char *>calloc(_ns[i], sizeof(unsigned char))
                memset(_masks[i], 1, _ns[i] * sizeof(unsigned char))
                ldims[0] = _ns[i]

                if group:
                    for j in range(n_group):
                        fail |= group_lines(_outs[i], _masks[i], _img + i * _Y * _X, _Y, _X, _outs[i],
                                            ldims, cutoff, group_threshold, dilation)

                if filter:
                    fail |= filter_lines(_outs[i], _masks[i], _img + i * _Y * _X, _Y, _X, _outs[i],
                                         ldims, filter_threshold, dilation)

            free(ldims)

        if fail:
            raise RuntimeError("LSD execution finished with an error")

        for i in range(repeats):
            out_dims[0] = _ns[i]
            streaks = ArrayWrapper.from_ptr(<void *>_outs[i]).to_ndarray(2, out_dims, np.NPY_FLOAT32)
            cond = ArrayWrapper.from_ptr(<void *>_masks[i]).to_ndarray(1, out_dims, np.NPY_BOOL)
            line_dict[i] = np.PyArray_Compress(streaks, cond, 0, <np.ndarray>NULL)

        out_dict['lines'] = line_dict

        if return_labels:
            for i in range(repeats):
                out_dims[0] = _reg_ys[i]
                out_dims[1] = _reg_xs[i]
                reg_dict[i] = ArrayWrapper.from_ptr(<void *>_regs[i]).to_ndarray(2, out_dims, np.NPY_INT32)
            
            out_dict['labels'] = reg_dict
        else:
            for i in range(repeats):
                free(_regs[i])

        free(_outs); free(_ns); free(_regs); free(_reg_xs); free(_reg_ys)

        return out_dict
