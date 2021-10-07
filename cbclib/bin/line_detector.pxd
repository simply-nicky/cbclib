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