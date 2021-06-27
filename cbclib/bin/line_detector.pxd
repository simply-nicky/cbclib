cdef extern from "lsd.h":
    int LineSegmentDetection(double **out, int *n_out, double *img, int img_x, int img_y,
                             double scale, double sigma_scale,
                             double quant, double ang_th, double log_eps,
                             double density_th, int n_bins,
                             int **reg_img, int *reg_x, int *reg_y) nogil