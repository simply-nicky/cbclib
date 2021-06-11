#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
cdef extern from "lsd.h":
    double *LineSegmentDetection(int *n_out, double *img, int X, int Y,
                                 double scale, double sigma_scale,
                                 double quant, double ang_th, double log_eps,
                                 double density_th, int n_bins,
                                 int **reg_img, int *reg_x, int *reg_y) nogil