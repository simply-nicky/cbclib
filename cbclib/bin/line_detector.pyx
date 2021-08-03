cimport numpy as np
import numpy as np
from libc.stdlib cimport free, malloc, calloc
from cpython.ref cimport Py_INCREF

# Numpy must be initialized. When using numpy from C or Cython you must
# *ALWAYS* do that, or you will have segfaults
np.import_array()

DEF QUANT = 2.0
DEF ANG_TH = 22.5
DEF DENSITY_TH = 0.7
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
    cdef double _scale
    cdef double _sigma_scale
    cdef double _log_eps

    def __cinit__(self, double scale=0.8, double sigma_scale=0.6, double log_eps=0.):
        self._scale = scale
        self._sigma_scale = sigma_scale
        self._log_eps = log_eps

    def __init__(self, scale=0.8, sigma_scale=0.6, log_eps=0):
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
        """

    @staticmethod
    cdef np.ndarray _scale_image(np.ndarray image):
    """LSD works with digital values in the range [0, 255]."""
        cdef double *data = <double *>np.PyArray_DATA(image)
        cdef np.npy_intp size = np.PyArray_SIZE(image)
        cdef int i
        cdef double _min = data[0]
        cdef double _max = data[0]
        for i in range(size):
            if data[i] > _max:
                _max = data[i]
            if data[i] < _min:
                _min = data[i]
        for i in range(size):
            data[i] = (data[i] - _min) / (_max - _min) * 255.
        return image

    @staticmethod
    cdef np.ndarray _check_image(np.ndarray image):
        cdef int ndim = image.ndim
        if ndim != 2:
            raise ValueError('Image must be a 2D array.')
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
        image = LSD._check_image(image)
        image = LSD._scale_image(image)

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
            fail =  LineSegmentDetection(&_out, &_n_out, _img, _img_x, _img_y,
                                         self._scale, self._sigma_scale, QUANT,
                                         ANG_TH, self._log_eps, DENSITY_TH, N_BINS,
                                         &_reg_img, &_reg_x, &_reg_y)

        if fail:
            raise RuntimeError("LSD execution finished with an error.")
        
        cdef np.npy_intp *out_dims = [_n_out, LINE_SIZE]
        cdef np.ndarray out = ArrayWrapper.from_ptr(<void *>_out).to_ndarray(2, out_dims, np.NPY_FLOAT64)

        cdef np.npy_intp *reg_dims = [_reg_y, _reg_x,]
        cdef np.ndarray reg_img = ArrayWrapper.from_ptr(<void *>_reg_img).to_ndarray(2, reg_dims, np.NPY_INT32)

        return {'lines': out, 'labels': reg_img}
