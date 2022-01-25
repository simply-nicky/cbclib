import numpy as np
from libc.stdlib cimport free, malloc, calloc
from cython.parallel import prange
from .image_proc cimport check_array

# Numpy must be initialized. When using numpy from C or Cython you must
# *ALWAYS* do that, or you will have segfaults
np.import_array()

DEF N_BINS = 1024
DEF LINE_SIZE = 7
DEF NOT_DEF = -1.0

cdef class ArrayWrapper:
    """A wrapper class for a C data structure. """

    def __cinit__(self):
        self._data = NULL

    def __dealloc__(self):
        if not self._data is NULL:
            free(self._data)
            self._data = NULL

cdef class LSD:
    """LSD  is a class for performing the streak detection
    on digital images with Line Segment Detector algorithm [LSD]_.

    References
    ----------
    .. [LSD] "LSD: a Line Segment Detector" by Rafael Grompone von Gioi,
             Jeremie Jakubowicz, Jean-Michel Morel, and Gregory Randall,
             Image Processing On Line, 2012. DOI:10.5201/ipol.2012.gjmr-lsd
             http://dx.doi.org/10.5201/ipol.2012.gjmr-lsd
    """

    def __cinit__(self, double scale=0.8, double sigma_scale=0.6, double log_eps=0.,
                  double ang_th=45.0, double density_th=0.7, double quant=2.0,
                  double x_c=NOT_DEF, double y_c=NOT_DEF):
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
        self.x_c = x_c
        self.y_c = y_c

    def __init__(self, scale=0.8, sigma_scale=0.6, log_eps=0, ang_th=45.0,
                 density_th=0.7, quant=2.0, double x_c=NOT_DEF, double y_c=NOT_DEF):
        """Create a LSD object for streak detection on digital images.

        Parameters
        ----------
        scale : float, optional
            When different from 1.0, LSD will scale the input image
            by 'scale' factor by Gaussian filtering, before detecting
            line segments. Default value is 0.8.
        sigma_scale : float, optional
            When `scale` is different from 1.0, the sigma of the Gaussian
            filter is :code:`sigma = sigma_scale / scale`, if scale is less
            than 1.0, and :code:`sigma = sigma_scale` otherwise. Default
            value is 0.6.
        log_eps : float, optional
            Detection threshold, accept if -log10(NFA) > log_eps.
            The larger the value, the more strict the detector is, and will
            result in less detections. The value -log10(NFA) is equivalent
            but more intuitive than NFA:

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

    def detect(self, np.ndarray image, double radius=1.0, bint filter_lines=False,
               bint return_labels=False, unsigned int num_threads=1):
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
        if image.ndim < 2:
            raise ValueError('Image must be a 2D array.')
        image = check_array(image, np.NPY_FLOAT64)

        cdef int ndim = image.ndim
        cdef double *_img = <double *>np.PyArray_DATA(image)
        cdef int _X = <int>image.shape[ndim - 1]
        cdef int _Y = <int>image.shape[ndim - 2]
        cdef int repeats = image.size / _X / _Y
        cdef np.ndarray streaks

        cdef double **_outs = <double **>malloc(repeats * sizeof(double *))
        if _outs is NULL:
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

        cdef int fail = 0, i
        cdef dict line_dict = {}, reg_dict = {}, out_dict = {}
        cdef np.npy_intp *out_dims = [0, LINE_SIZE]

        num_threads = repeats if <int>num_threads > repeats else <int>num_threads

        if filter_lines and self.x_c > 0.0 and self.x_c < _X and self.y_c > 0 and self.y_c < _Y:
            for i in prange(repeats, schedule='guided', num_threads=num_threads, nogil=True):
                fail |= LineSegmentDetection(&_outs[i], &_ns[i], _img + i * _Y * _X, _Y, _X,
                                            self.scale, self.sigma_scale, self.quant,
                                            self.ang_th, self.log_eps, self.density_th, N_BINS,
                                            &_regs[i], &_reg_ys[i], &_reg_xs[i])
                fail |= filter_lines_c(_outs[i], _img + i * _Y * _X, _Y, _X, _outs[i], _ns[i],
                                       self.x_c, self.y_c, radius)
        else:
            for i in prange(repeats, schedule='guided', num_threads=num_threads, nogil=True):
                fail |= LineSegmentDetection(&_outs[i], &_ns[i], _img + i * _Y * _X, _Y, _X,
                                            self.scale, self.sigma_scale, self.quant,
                                            self.ang_th, self.log_eps, self.density_th, N_BINS,
                                            &_regs[i], &_reg_ys[i], &_reg_xs[i])

        if fail:
            raise RuntimeError("LSD execution finished with an error")

        for i in range(repeats):
            out_dims[0] = _ns[i]
            streaks = ArrayWrapper.from_ptr(<void *>_outs[i]).to_ndarray(2, out_dims, np.NPY_FLOAT64)
            line_dict[i] = np.PyArray_Compress(streaks, streaks[:, 0], 0, <np.ndarray>NULL)

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

    def mask(self, np.ndarray image, unsigned int max_val=1, unsigned int dilation=0,
             double radius=1.0, bint filter_lines=False, bint return_lines=True,
             unsigned int num_threads=1):
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
        if image.ndim < 2:
            raise ValueError('Image must be >=2D array.')
        image = LSD._check_image(image)

        cdef int ndim = image.ndim
        cdef double *_img = <double *>np.PyArray_DATA(image)
        cdef int _X = <int>image.shape[ndim - 1]
        cdef int _Y = <int>image.shape[ndim - 2]

        cdef np.ndarray streaks
        cdef np.ndarray mask = np.PyArray_ZEROS(ndim, image.shape, np.NPY_UINT32, 0)
        cdef unsigned int *msk_ptr = <unsigned int *>np.PyArray_DATA(mask)
        cdef int repeats = image.size / _X / _Y

        cdef double **_outs = <double **>malloc(repeats * sizeof(double *))
        if _outs is NULL:
            raise MemoryError('not enough memory')

        cdef int *_ns = <int *>malloc(repeats * sizeof(int))
        if _ns is NULL:
            free(_outs)
            raise MemoryError('not enough memory')

        cdef int fail = 0, i
        cdef dict line_dict = {}, out_dict = {}
        cdef np.npy_intp *out_dims = [0, LINE_SIZE]

        num_threads = repeats if <int>num_threads > repeats else <int>num_threads

        if filter_lines and self.x_c > 0.0 and self.x_c < _X and self.y_c > 0.0 and self.y_c < _Y:
            for i in prange(repeats, schedule='guided', num_threads=num_threads, nogil=True):
                fail |= LineSegmentDetection(&_outs[i], &_ns[i], _img + i * _Y * _X, _Y, _X,
                                             self.scale, self.sigma_scale, self.quant,
                                             self.ang_th, self.log_eps, self.density_th, N_BINS,
                                             NULL, NULL, NULL)
                fail |= filter_lines_c(_outs[i], _img + i * _Y * _X, _Y, _X, _outs[i], _ns[i],
                                       self.x_c, self.y_c, radius)
                draw_lines(msk_ptr + i * _Y * _X, _Y, _X, max_val, _outs[i], _ns[i], dilation)
        else:
            for i in prange(repeats, schedule='guided', num_threads=num_threads, nogil=True):
                fail |= LineSegmentDetection(&_outs[i], &_ns[i], _img + i * _Y * _X, _Y, _X,
                                             self.scale, self.sigma_scale, self.quant,
                                             self.ang_th, self.log_eps, self.density_th, N_BINS,
                                             NULL, NULL, NULL)
                draw_lines(msk_ptr + i * _Y * _X, _Y, _X, max_val, _outs[i], _ns[i], dilation)

        if fail:
            raise RuntimeError("LSD execution finished with an error")

        out_dict['mask'] = mask

        if return_lines:
            for i in range(repeats):
                out_dims[0] = _ns[i]
                streaks = ArrayWrapper.from_ptr(<void *>_outs[i]).to_ndarray(2, out_dims, np.NPY_FLOAT64)
                line_dict[i] = np.PyArray_Compress(streaks, streaks[:, 0], 0, <np.ndarray>NULL)

            out_dict['lines'] = line_dict
        else:
            for i in range(repeats):
                free(_outs[i])

        free(_ns); free(_outs)

        return out_dict