cimport numpy as np
import numpy as np
from libc.stdlib cimport free, malloc, calloc
from cpython.ref cimport Py_INCREF
from cython.parallel import prange

# Numpy must be initialized. When using numpy from C or Cython you must
# *ALWAYS* do that, or you will have segfaults
np.import_array()

DEF N_BINS = 1024
DEF LINE_SIZE = 7

cdef class ArrayWrapper:
    """A wrapper class for a C data structure. """
    cdef void* _data

    def __cinit__(self):
        self._data = NULL

    def __dealloc__(self):
        if not self._data is NULL:
            free(self._data)
            self._data = NULL

    @staticmethod
    cdef ArrayWrapper from_ptr(void *data):
        """Factory function to create a new wrapper from a C pointer."""
        cdef ArrayWrapper wrapper = ArrayWrapper.__new__(ArrayWrapper)
        wrapper._data = data
        return wrapper

    cdef np.ndarray to_ndarray(self, int ndim, np.npy_intp *dims, int type_num):
        """Get a NumPy array from a wrapper."""
        cdef np.ndarray ndarray = np.PyArray_SimpleNewFromData(ndim, dims, type_num, self._data)

        # without this, data would be cleaned up right away
        Py_INCREF(self)
        np.PyArray_SetBaseObject(ndarray, self)
        return ndarray

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
    cdef public double ang_th
    cdef public double density_th
    cdef public double log_eps
    cdef public double scale
    cdef public double sigma_scale
    cdef public double quant

    def __cinit__(self, double scale=0.8, double sigma_scale=0.6, double log_eps=0.,
                  double ang_th=45.0, double density_th=0.7, double quant=2.0):
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

    def __init__(self, scale=0.8, sigma_scale=0.6, log_eps=0, ang_th=45.0, density_th=0.7, quant=2.0):
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

    @staticmethod
    cdef np.ndarray _check_image(np.ndarray image):
        if not np.PyArray_IS_C_CONTIGUOUS(image):
            image = np.PyArray_GETCONTIGUOUS(image)
        cdef int tn = np.PyArray_TYPE(image)
        if tn != np.NPY_FLOAT64:
            image = np.PyArray_Cast(image, np.NPY_FLOAT64)
        return image

    cpdef dict detect(self, np.ndarray image):
        """Perform the streak detection on `image`.

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
        if image.ndim != 2:
            raise ValueError('Image must be a 2D array.')
        image = LSD._check_image(image)

        cdef double *_img = <double *>np.PyArray_DATA(image)
        cdef int _img_x = <int>image.shape[1]
        cdef int _img_y = <int>image.shape[0]

        cdef int _reg_x
        cdef int _reg_y
        cdef int *_reg_img

        cdef int _n_out
        cdef double *_out

        cdef int fail = 0
        with nogil:
            fail = LineSegmentDetection(&_out, &_n_out, _img, _img_x, _img_y,
                                        self.scale, self.sigma_scale, self.quant,
                                        self.ang_th, self.log_eps, self.density_th, N_BINS,
                                        &_reg_img, &_reg_x, &_reg_y)

        if fail:
            raise RuntimeError("LSD execution finished with an error.")
        
        cdef np.npy_intp *out_dims = [_n_out, LINE_SIZE]
        cdef np.ndarray out = ArrayWrapper.from_ptr(<void *>_out).to_ndarray(2, out_dims, np.NPY_FLOAT64)

        cdef np.npy_intp *reg_dims = [_reg_y, _reg_x,]
        cdef np.ndarray reg_img = ArrayWrapper.from_ptr(<void *>_reg_img).to_ndarray(2, reg_dims, np.NPY_INT32)

        return {'lines': out, 'labels': reg_img}

    cpdef np.ndarray mask(self, np.ndarray image, unsigned int max_val=1, unsigned int dilation=0, unsigned int num_threads=1):
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
        cdef double *img_ptr = <double *>np.PyArray_DATA(image)
        cdef int X = <int>image.shape[ndim - 1]
        cdef int Y = <int>image.shape[ndim - 2]

        cdef np.ndarray mask = np.PyArray_ZEROS(ndim, image.shape, np.NPY_UINT32, 0)
        cdef unsigned int *msk_ptr = <unsigned int *>np.PyArray_DATA(mask)
        cdef double *out
        cdef int n_out, fail, i

        cdef unsigned int repeats = image.size / X / Y
        num_threads = repeats if num_threads > repeats else num_threads
        for i in prange(repeats, schedule='guided', num_threads=num_threads, nogil=True):
            fail = LineSegmentDetection(&out, &n_out, img_ptr + i * X * Y, X, Y,
                                        self.scale, self.sigma_scale, self.quant,
                                        self.ang_th, self.log_eps, self.density_th, N_BINS,
                                        NULL, NULL, NULL)
            draw_lines(msk_ptr + i * X * Y, X, Y, max_val, out, n_out, dilation)
        return mask