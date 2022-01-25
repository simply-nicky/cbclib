cimport numpy as np
from cpython.ref cimport Py_INCREF

cdef extern from "lsd.h":
    int LineSegmentDetection(double **out, int *n_out, double *img, int img_x, int img_y,
                             double scale, double sigma_scale,
                             double quant, double ang_th, double log_eps,
                             double density_th, int n_bins,
                             int **reg_img, int *reg_x, int *reg_y) nogil

cdef extern from "img_proc.h":
    int draw_lines(unsigned int *out, unsigned long Y, unsigned long X,
                   unsigned int max_val, double *lines, unsigned long lines,
                   unsigned int dilation) nogil

    int filter_lines_c "filter_lines" (double *olines, double *data, unsigned long Y,
                       unsigned long X, double *ilines, unsigned long n_lines, double x_c,
                       double y_c, double radius) nogil

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
    cdef public double ang_th
    cdef public double density_th
    cdef public double log_eps
    cdef public double scale
    cdef public double sigma_scale
    cdef public double quant
    cdef public double x_c
    cdef public double y_c