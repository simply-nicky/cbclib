import numpy as np
from cython.parallel import prange, parallel
<<<<<<< HEAD
=======
from .image_proc cimport check_array, normalize_sequence

cdef line_profile profiles[4]
cdef void build_profiles():
    profiles[0] = linear_profile
    profiles[1] = quad_profile
    profiles[2] = tophat_profile
    profiles[3] = gauss_profile

cdef dict profile_scheme
profile_scheme = {'linear': 0, 'quad': 1, 'tophat': 2, 'gauss': 3}

build_profiles()
>>>>>>> dev-dataclass

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
<<<<<<< HEAD
    """LSD  is a class for performing the streak detection
    on digital images with Line Segment Detector algorithm [LSD]_.

    References
    ----------
    .. [LSD] "LSD: a Line Segment Detector" by Rafael Grompone von Gioi,
             Jeremie Jakubowicz, Jean-Michel Morel, and Gregory Randall,
             Image Processing On Line, 2012. DOI:10.5201/ipol.2012.gjmr-lsd
             http://dx.doi.org/10.5201/ipol.2012.gjmr-lsd
    """

    def __cinit__(self, float scale=0.8, float sigma_scale=0.6, float log_eps=0.,
                  float ang_th=45.0, float density_th=0.7, float quant=2.0):
=======
    def __cinit__(self, float scale=0.9, float sigma_scale=0.9, float log_eps=0.0,
                  float ang_th=45.0, float density_th=0.7, float quant=2e-2):
>>>>>>> dev-dataclass
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

<<<<<<< HEAD
    def __init__(self, float scale=0.8, float sigma_scale=0.6, float log_eps=0,
                 float ang_th=45.0, float density_th=0.7, float quant=2.0):
        """Create a LSD object for streak detection on digital images.
=======
    def __str__(self):
        return self.state_dict().__str__()
>>>>>>> dev-dataclass

    def __repr__(self):
        return self.state_dict().__repr__()

<<<<<<< HEAD
            * -1.0 gives an average of 10 false detections on noise.
            *  0.0 gives an average of 1 false detections on noise.
            *  1.0 gives an average of 0.1 false detections on nose.
            *  2.0 gives an average of 0.01 false detections on noise.
            Default value is 0.0.
        ang_th : float, optional
            Gradient angle tolerance in the region growing algorithm, in
            degrees. Default value is 45.0.
        density_th : float, optional
            Minimal proportion of 'supporting' points in a rectangle.
            Default value is 0.7.
        quant : float, optional
            Bound to the quantization error on the gradient norm.
            Example: if gray levels are quantized to integer steps,
            the gradient (computed by finite differences) error
            due to quantization will be bounded by 2.0, as the
            worst case is when the error are 1 and -1, that
            gives an error of 2.0. Default value is 2.0.
        """

    def detect(self, np.ndarray image not None, float cutoff, float filter_threshold=0.0,
               float group_threshold=0.6, bint filter=True, bint group=True, int n_group=2,
               float dilation=6.0, bint return_labels=False, unsigned int num_threads=1):
        """Perform the LSD streak detection on `image`.

        Parameters
        ----------
        image : np.ndarray
            2D array of the digital image.
        
        Returns
        -------
        dict
            :class:`dict` with the following fields:

            * `lines` : An array of the detected lines. Each line is
            comprised of 7 parameters as follows:

                * `[x1, y1]`, `[x2, y2]` : The coordinates of the line's
                  ends.
                * `width` : Line's width.
                * `p` : Angle precision [0, 1] given by angle tolerance
                  over 180 degree.
                * `-log10(NFA)` : Number of false alarms.
            
            * `labels` : image where each pixel indicates the line
              segment to which it belongs. Unused pixels have the value
              0, while the used ones have the number of the line segment,
              numbered in the same order as in `lines`.
        """
=======
    def detect(self, np.ndarray image not None, float cutoff, float filter_threshold=0.0,
               float group_threshold=1.0, float dilation=0.0, str profile='linear',
               bint return_labels=False, unsigned int num_threads=1):
>>>>>>> dev-dataclass
        if image.ndim < 2:
            raise ValueError('Image must be a 2D array.')
        image = check_array(image, np.NPY_FLOAT32)

        cdef int ndim = image.ndim
        cdef float *_img = <float *>np.PyArray_DATA(image)
<<<<<<< HEAD
        cdef int _X = <int>image.shape[ndim - 1]
        cdef int _Y = <int>image.shape[ndim - 2]
        cdef int repeats = image.size / _X / _Y
=======
        cdef unsigned long *_dims = <unsigned long *>image.shape + ndim - 2
        cdef int repeats = image.size / _dims[0] / _dims[1]
>>>>>>> dev-dataclass
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

        cdef line_profile _prof = profiles[profile_scheme[profile]]
        num_threads = repeats if <int>num_threads > repeats else <int>num_threads
        cdef float *buf
        cdef float *vmin
        cdef float *vmax

        with nogil, parallel(num_threads=num_threads):
            ldims = <unsigned long *>malloc(2 * sizeof(unsigned long))
            ldims[1] = LINE_SIZE
<<<<<<< HEAD

            for i in prange(repeats, schedule='guided'):
                fail |= LineSegmentDetection(&_outs[i], &_ns[i], _img + i * _Y * _X, _Y, _X,
                                            self.scale, self.sigma_scale, self.quant,
                                            self.ang_th, self.log_eps, self.density_th, N_BINS,
                                            &_regs[i], &_reg_ys[i], &_reg_xs[i])
=======
            buf = <float *>malloc(_dims[0] * _dims[1] * sizeof(float))

            for i in prange(repeats, schedule='guided'):
                memcpy(buf, _img + i * _dims[0] * _dims[1], _dims[0] * _dims[1] * sizeof(float))
                vmin = <float *>wirthselect(buf, 0, _dims[0] * _dims[1], sizeof(float), compare_float)
                vmax = <float *>wirthselect(buf, _dims[0] * _dims[1] - 1, _dims[0] * _dims[1], sizeof(float),
                                            compare_float)

                fail |= LineSegmentDetection(&_outs[i], &_ns[i], _img + i * _dims[0] * _dims[1], _dims,
                                             self.scale, self.sigma_scale, self.quant * (vmax[0] - vmin[0]),
                                             self.ang_th, self.log_eps, self.density_th, N_BINS,
                                             &_regs[i], &_reg_ys[i], &_reg_xs[i])

                _masks[i] = <unsigned char *>calloc(_ns[i], sizeof(unsigned char))
                memset(_masks[i], 1, _ns[i] * sizeof(unsigned char))
                ldims[0] = _ns[i]

                fail |= group_line(_outs[i], _masks[i], _img + i * _dims[0] * _dims[1], _dims, _outs[i],
                                   ldims, cutoff, group_threshold, dilation, _prof)

                fail |= filter_line(_outs[i], _masks[i], _img + i * _dims[0] * _dims[1], _dims, _outs[i],
                                    ldims, filter_threshold, dilation, _prof)

            free(ldims)
            free(buf)
>>>>>>> dev-dataclass

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

<<<<<<< HEAD
    def draw_lines(self, np.ndarray mask not None, dict lines not None,
                   int max_val=1, double dilation=0.0, unsigned int num_threads=1):
        """Perform the streak detection on `image` and return rasterized lines
        drawn on a mask array.

        Parameters
        ----------
        image : np.ndarray
            2D array of the digital image.
        max_val : int, optional
            Maximal value in the output mask.
        dilation : int, optional
            Size of the morphology dilation applied to the output mask.
        num_threads : int, optional
            Number of the computational threads.
        
        Returns
        -------
        mask : np.ndarray
            Array, that has the same shape as `image`, with the regions
            masked by the detected lines.
        """
        if mask.ndim < 2:
            raise ValueError('Mask must be >=2D array.')
        mask = check_array(mask, np.NPY_UINT32)

        cdef int ndim = mask.ndim
        cdef unsigned int *_mask = <unsigned int *>np.PyArray_DATA(mask)
        cdef int _X = <int>mask.shape[ndim - 1]
        cdef int _Y = <int>mask.shape[ndim - 2]
        cdef int repeats = mask.size / _X / _Y

        cdef int fail = 0, i, N = len(lines)
        cdef list frames = list(lines)
        cdef float **_lines = <float **>malloc(N * sizeof(float *))
        cdef unsigned long **_ldims = <unsigned long **>malloc(N * sizeof(unsigned long *))
        cdef np.ndarray _larr
        for i in range(N):
            _larr = lines[frames[i]]
            _lines[i] = <float *>np.PyArray_DATA(_larr)
            _ldims[i] = <unsigned long *>_larr.shape

        if N < repeats:
            repeats = N
        num_threads = repeats if <int>num_threads > repeats else <int>num_threads        

        for i in prange(repeats, schedule='guided', num_threads=num_threads, nogil=True):
            draw_lines(_mask + i * _Y * _X, _Y, _X, max_val, _lines[i], _ldims[i], <float>dilation)

        if fail:
            raise RuntimeError("LSD execution finished with an error")

        free(_lines); free(_ldims)

        return mask
=======
    def state_dict(self):
        return {'ang_th': self.ang_th, 'density_th': self.density_th, 'log_eps': self.log_eps,
                'scale': self.scale, 'sigma_scale': self.sigma_scale, 'quant': self.quant}
>>>>>>> dev-dataclass
