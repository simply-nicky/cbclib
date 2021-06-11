#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
cimport numpy as np
import numpy as np

# Numpy must be initialized. When using numpy from C or Cython you must
# *ALWAYS* do that, or you will have segfaults
np.import_array()

DEF QUANT = 2.0
DEF ANG_TH = 22.5
DEF DENSITY_TH = 0.7

class LSD:
    double _scale
    double _sigma_scale
    double _log_eps

    def __cinit__(self, double scale=0.8, double sigma_scale=0.6, double log_eps=0.):
        self._img = NULL
        self._img_x = NULL
        self._ing_y = NULL

        self._reg_img = NULL
        self._reg_x = NULL
        self._reg_y = NULL

        self._n_out = NULL
        self._out = NULL

        self._scale = scale
        self._sigma_scale = sigma_scale
        self._log_eps = log_eps

    def __init__(self, scale=0.8, sigma_scale=0.6, log_eps=0):
        """Line Segment Detector.
        """
    
    def __dealloc__(self):
        if not self._img_x == NULL:
            free(self._img_x)
        
        if not self._img_y == NULL:
            free(self._img_y)

        if not self._reg_x == NULL:
            free(self._reg_x)
        
        if not self._reg_y == NULL:
            free(self._reg_y)
        
        if not self._n_out == NULL:
            free(self._n_out)

    cdef np.ndarray _check_image(np.ndarray image):
        cdef int ndim
        if not np.PyArray_IS_C_CONTIGUOUS(image):
            image = np.PyArray_GETCONTIGUOUS(image)
        cdef int tn = np.PyArray_TYPE(array)
        if tn != np.NPY_FLOAT64:
            image = np.PyArray_Cast(array, np.NPY_FLOAT64)
        return image

    cpdef dict detect(self, np.ndarray image):


        cdef double *_img
        cdef int *_img_x
        cdef int *_img_y
        cdef int *_reg_img
        cdef int *_reg_x
        cdef int *_reg_y

        cdef int *_n_out
        cdef double *_out    
    