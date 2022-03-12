cimport numpy as np
from cpython.ref cimport Py_INCREF

cdef extern from "lsd.h":
    int LineSegmentDetection(float **out, int *n_out, float *img, int img_x, int img_y,
                             float scale, float sigma_scale,
                             float quant, float ang_th, float log_eps,
                             float density_th, int n_bins,
                             int **reg_img, int *reg_x, int *reg_y) nogil

cdef extern from "img_proc.h":
    int draw_lines(unsigned int *out, unsigned long Y, unsigned long X,
                   unsigned int max_val, float *lines, unsigned long lines,
                   unsigned int dilation) nogil

    int filter_lines_c "filter_lines" (float *olines, float *data, unsigned long Y,
                       unsigned long X, float *ilines, unsigned long n_lines, float x_c,
                       float y_c, float *radii, float thr) nogil

cdef class ArrayWrapper:
    cdef void* _data

    @staticmethod
    cdef inline ArrayWrapper from_ptr(void *data):
        """Factory function to create a new wrapper from a C pointer."""
        cdef ArrayWrapper wrapper = ArrayWrapper.__new__(ArrayWrapper)
        wrapper._data = data
        return wrapper

    cdef inline np.ndarray to_ndarray(self, int ndim, np.npy_intp *dims, int type_num):
        """Get a NumPy array from a wrapper."""
        cdef np.ndarray ndarray = np.PyArray_SimpleNewFromData(ndim, dims, type_num, self._data)

        # without this, data would be cleaned up right away
        Py_INCREF(self)
        np.PyArray_SetBaseObject(ndarray, self)
        return ndarray

cdef class LSD:
    cdef public float ang_th
    cdef public float density_th
    cdef public float log_eps
    cdef public float scale
    cdef public float sigma_scale
    cdef public float quant
    cdef public float x_c
    cdef public float y_c