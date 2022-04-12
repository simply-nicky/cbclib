#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
import numpy as np
import cython
from math import ceil
from cython.parallel import parallel, prange


# Numpy must be initialized. When using numpy from C or Cython you must
# *ALWAYS* do that, or you will have segfaults
np.import_array()

DEF N_BINS = 1024
DEF LINE_SIZE = 7

def project_effs(np.ndarray data not None, np.ndarray mask not None, np.ndarray effs not None,
                 np.ndarray out=None, int num_threads=1) -> np.ndarray:
    data = check_array(data, np.NPY_FLOAT32)
    mask = check_array(mask, np.NPY_BOOL)
    effs = check_array(effs, np.NPY_FLOAT32)

    cdef int i, j, k, ii
    cdef np.float32_t w1, w0

    if out is None:
        out = <np.ndarray>np.PyArray_ZEROS(data.ndim, data.shape, np.NPY_FLOAT32, 0)

    cdef np.float32_t[:, :, ::1] _data = data
    cdef np.npy_bool[:, :, ::1] _mask = mask
    cdef np.float32_t[:, :, ::1] _effs = effs
    cdef np.float32_t[:, :, ::1] _out = out

    cdef int n_frames = _data.shape[0]
    num_threads = n_frames if <int>num_threads > n_frames else <int>num_threads
    for i in prange(n_frames, schedule='guided', nogil=True):
        for ii in range(_effs.shape[0]):
            w1 = 0.0; w0 = 0.0
            for j in range(_data.shape[1]):
                for k in range(_data.shape[2]):
                    if _mask[i, j, k]:
                        w1 = w1 + _data[i, j, k] * _effs[ii, j, k]
                        w0 = w0 + _effs[ii, j, k] * _effs[ii, j, k]
            w1 = w1 / w0 if w0 > 0.0 else 1.0
            for j in range(_data.shape[1]):
                for k in range(_data.shape[2]):
                    _out[i, j, k] = _out[i, j, k] + _effs[ii, j, k] * w1

    return out

def subtract_background(np.ndarray data not None, np.ndarray mask not None, np.ndarray whitefields not None,
                        int num_threads=1) -> np.ndarray:
    data = check_array(data, np.NPY_UINT32)
    mask = check_array(mask, np.NPY_BOOL)
    whitefields = check_array(whitefields, np.NPY_FLOAT32)

    cdef int i, j, k
    cdef float res, w0, w1

    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(data.ndim, data.shape, np.NPY_FLOAT32)

    cdef np.uint32_t[:, :, ::1] _data = data
    cdef np.npy_bool[:, :, ::1] _mask = mask
    cdef np.float32_t[:, :, ::1] _out = out
    cdef np.float32_t[:, :, ::1] _whitefields = whitefields

    cdef int n_frames = _data.shape[0]
    num_threads = n_frames if <int>num_threads > n_frames else <int>num_threads

    for i in prange(n_frames, schedule='guided', num_threads=num_threads, nogil=True):
        for j in range(_data.shape[1]):
            for k in range(_data.shape[2]):
                if _mask[i, j, k]:
                    res = <float>_data[i, j, k] - _whitefields[i, j, k]
                    _out[i, j, k] = res if res > 0.0 else 0.0
                else:
                    _out[i, j, k] = 0.0

    return out

def median_filter(np.ndarray data not None, object size=None, np.ndarray footprint=None,
                  np.ndarray mask=None, np.ndarray good_data=None, str mode='reflect', double cval=0.0,
                  unsigned num_threads=1):
    """Calculate a median along the `axis`.

    Args:
        data (numpy.ndarray) : Intensity frames.
        size (Optional[Union[int, Tuple[int, ...]]]) : See footprint, below. Ignored if footprint
            is given.
        footprint (Optional[numpy.ndarray]) :  Either size or footprint must be defined. size
            gives the shape that is taken from the input array, at every element position, to
            define the input to the filter function. footprint is a boolean array that specifies
            (implicitly) a shape, but also which of the elements within this shape will get passed
            to the filter function. Thus size=(n,m) is equivalent to footprint=np.ones((n,m)).
            We adjust size to the number of dimensions of the input array, so that, if the input
            array is shape (10,10,10), and size is 2, then the actual size used is (2,2,2). When
            footprint is given, size is ignored.
        mask (Optional[numpy.ndarray]) : Bad pixel mask.
        mode (str) : The mode parameter determines how the input array is extended when the
            filter overlaps a border. Default value is 'reflect'. The valid values and their
            behavior is as follows:

            * `constant`, (k k k k | a b c d | k k k k) : The input is extended by filling all
              values beyond the edge with the same constant value, defined by the `cval`
              parameter.
            * `nearest`, (a a a a | a b c d | d d d d) : The input is extended by replicating
              the last pixel.
            * `mirror`, (c d c b | a b c d | c b a b) : The input is extended by reflecting
              about the center of the last pixel. This mode is also sometimes referred to as
              whole-sample symmetric.
            * `reflect`, (d c b a | a b c d | d c b a) : The input is extended by reflecting
              about the edge of the last pixel. This mode is also sometimes referred to as
              half-sample symmetric.
            * `wrap`, (a b c d | a b c d | a b c d) : The input is extended by wrapping around
              to the opposite edge.
        cval (float) : Value to fill past edges of input if mode is 'constant'. Default is 0.0.
        num_threads (int) : Number of threads used in the calculations.

    Returns:
        numpy.ndarray : Filtered array. Has the same shape as `input`.
    """
    if not np.PyArray_IS_C_CONTIGUOUS(data):
        data = np.PyArray_GETCONTIGUOUS(data)

    cdef int ndim = data.ndim
    cdef np.npy_intp *dims = data.shape

    if mask is None:
        mask = <np.ndarray>np.PyArray_SimpleNew(ndim, dims, np.NPY_BOOL)
        np.PyArray_FILLWBYTE(mask, 1)
    else:
        mask = check_array(mask, np.NPY_BOOL)

    if good_data is None:
        good_data = mask
    else:
        good_data = check_array(good_data, np.NPY_BOOL)

    if size is None and footprint is None:
        raise ValueError('size or footprint must be provided.')

    cdef unsigned long *_fsize
    cdef np.ndarray fsize
    if size is None:
        _fsize = <unsigned long *>footprint.shape
    else:
        fsize = normalize_sequence(size, ndim, np.NPY_INTP)
        _fsize = <unsigned long *>np.PyArray_DATA(fsize)

    if footprint is None:
        footprint = <np.ndarray>np.PyArray_SimpleNew(ndim, <np.npy_intp *>_fsize, np.NPY_BOOL)
        np.PyArray_FILLWBYTE(footprint, 1)
    cdef unsigned char *_fmask = <unsigned char *>np.PyArray_DATA(footprint)

    cdef unsigned long *_dims = <unsigned long *>dims
    cdef int type_num = np.PyArray_TYPE(data)
    cdef np.ndarray out = <np.ndarray>np.PyArray_SimpleNew(ndim, dims, type_num)
    cdef void *_out = <void *>np.PyArray_DATA(out)
    cdef void *_data = <void *>np.PyArray_DATA(data)
    cdef unsigned char *_mask = <unsigned char *>np.PyArray_DATA(mask)
    cdef unsigned char *_gdata = <unsigned char *>np.PyArray_DATA(good_data)
    cdef int _mode = extend_mode_to_code(mode)
    cdef void *_cval = <void *>&cval

    with nogil:
        if type_num == np.NPY_FLOAT64:
            fail = median_filter_c(_out, _data, _mask, _gdata, ndim, _dims, 8, _fsize, _fmask, _mode, _cval, compare_double, num_threads)
        elif type_num == np.NPY_FLOAT32:
            fail = median_filter_c(_out, _data, _mask, _gdata, ndim, _dims, 4, _fsize, _fmask, _mode, _cval, compare_float, num_threads)
        elif type_num == np.NPY_INT32:
            fail = median_filter_c(_out, _data, _mask, _gdata, ndim, _dims, 4, _fsize, _fmask, _mode, _cval, compare_int, num_threads)
        elif type_num == np.NPY_UINT32:
            fail = median_filter_c(_out, _data, _mask, _gdata, ndim, _dims, 4, _fsize, _fmask, _mode, _cval, compare_uint, num_threads)
        elif type_num == np.NPY_UINT64:
            fail = median_filter_c(_out, _data, _mask, _gdata, ndim, _dims, 8, _fsize, _fmask, _mode, _cval, compare_ulong, num_threads)
        else:
            raise TypeError('data argument has incompatible type: {:s}'.format(data.dtype))
    if fail:
        raise RuntimeError('C backend exited with error.')

    return out

def draw_lines_aa(np.ndarray image not None, np.ndarray lines not None, int max_val=255, double dilation=0.0) -> np.ndarray:
    """Draw thick lines with variable thickness and the antialiasing applied.
    The lines must follow the LSD convention, see the parameters for more info.

    Args:
        image (numpy.ndarray) : Image array.
        lines (numpy.ndarray) : An array of the detected lines. Must have a shape
            of (`N`, 7), where `N` is the number of lines. Each line is comprised
            of 7 parameters as follows:

            * `[x1, y1]`, `[x2, y2]` : The coordinates of the line's
            ends.
            * `width` : Line's width.
            * `p` : Angle precision [0, 1] given by angle tolerance
            over 180 degree.
            * `-log10(NFA)` : Number of false alarms.

        max_val (int) : Maximum value of the line mask.
        dilation (int) : Size of the binary dilation applied to the output image.

    Returns:
        numpy.ndarray : Output image with the lines drawn.

    See Also:
        :class:`cbclib.bin.LSD` : Line Segment Detector.
    """
    image = check_array(image, np.NPY_UINT32)
    lines = check_array(lines, np.NPY_FLOAT32)

    if image.ndim != 2:
        raise ValueError("image array must be two-dimensional")
    if lines.ndim != 2 or lines.shape[1] < 5 or lines.shape[1] > 7:
        raise ValueError(f"lines array has an incompatible shape")

    cdef unsigned int *_image = <unsigned int *>np.PyArray_DATA(image)
    cdef unsigned long _Y = image.shape[0]
    cdef unsigned long _X = image.shape[1]
    cdef float *_lines = <float *>np.PyArray_DATA(lines)
    cdef unsigned long *_ldims = <unsigned long *>lines.shape

    with nogil:
        fail = draw_lines(_image, _Y, _X, max_val, _lines, _ldims, <float>dilation)
    if fail:
        raise RuntimeError('C backend exited with error.')    
    return image

def draw_line_indices_aa(np.ndarray lines not None, object shape not None, int max_val=255, double dilation=0.0) -> np.ndarray:
    """Draw thick lines with variable thickness and the antialiasing applied.
    The lines must follow the LSD convention, see the parameters for more info.

    Args:
        lines (numpy.ndarray) : An array of the detected lines. Must have a shape
            of (`N`, 7), where `N` is the number of lines. Each line is comprised
            of 7 parameters as follows:

            * `[x1, y1]`, `[x2, y2]` : The coordinates of the line's
            ends.
            * `width` : Line's width.
            * `p` : Angle precision [0, 1] given by angle tolerance
            over 180 degree.
            * `-log10(NFA)` : Number of false alarms.

        shape (Iterable[int]) : Shape of the image.
        max_val (int) : Maximum value of the line mask.
        dilation (int) : Size of the binary dilation applied to the output image.

    Returns:
        numpy.ndarray : Output line indices.

    See Also:
        :class:`cbclib.bin.LSD` : Line Segment Detector.
    """
    lines = check_array(lines, np.NPY_FLOAT32)

    if lines.ndim != 2 or lines.shape[1] < 5 or lines.shape[1] > 7:
        raise ValueError(f"lines array has an incompatible shape")

    cdef np.ndarray _shape = normalize_sequence(shape, 2, np.NPY_INTP)
    cdef unsigned long _Y = _shape[0]
    cdef unsigned long _X = _shape[1]

    cdef unsigned int *_idxs
    cdef unsigned long _n_idxs
    cdef float *_lines = <float *>np.PyArray_DATA(lines)
    cdef unsigned long *_ldims = <unsigned long *>lines.shape

    with nogil:
        fail = draw_line_indices(&_idxs, &_n_idxs, _Y, _X, max_val, _lines, _ldims, <float>dilation)
    if fail:
        raise RuntimeError('C backend exited with error.')    

    cdef np.npy_intp *dim = [_n_idxs, 4]
    cdef np.ndarray idxs = ArrayWrapper.from_ptr(<void *>_idxs).to_ndarray(2, dim, np.NPY_UINT32)

    return idxs

def tilt_matrix(tilts: np.ndarray, axis: object) -> np.ndarray:
    cdef np.ndarray _axis = normalize_sequence(axis, 3, np.NPY_FLOAT64)

    cdef np.npy_intp *rmdims = [tilts.shape[0], 3, 3]
    cdef np.ndarray rot_mats = <np.ndarray>np.PyArray_SimpleNew(3, rmdims, np.NPY_FLOAT64)

    cdef double *t_ptr = <double *>np.PyArray_DATA(tilts)
    cdef double *rm_ptr = <double *>np.PyArray_DATA(rot_mats)
    cdef unsigned long n_mats = tilts.shape[0]
    cdef double a0 = _axis[0], a1 = _axis[1], a2 = _axis[2]

    cdef int fail = 0
    with nogil:
        fail = generate_rot_matrix(rm_ptr, t_ptr, n_mats, a0, a1, a2)
    if fail:
        raise RuntimeError('C backend exited with error.')
        
    return rot_mats

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

    def __cinit__(self, float scale=0.8, float sigma_scale=0.6, float log_eps=0.,
                  float ang_th=45.0, float density_th=0.7, float quant=2.0):
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

    def __init__(self, float scale=0.8, float sigma_scale=0.6, float log_eps=0,
                 float ang_th=45.0, float density_th=0.7, float quant=2.0):
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

    def detect(self, np.ndarray image not None, float cutoff, float filter_threshold=0.0,
               float group_threshold=0.6, bint filter=True, bint group=True, int n_group=2,
               float dilation=6.0,  bint return_labels=False, unsigned int num_threads=1):
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
        image = check_array(image, np.NPY_FLOAT32)

        cdef int ndim = image.ndim
        cdef float *_img = <float *>np.PyArray_DATA(image)
        cdef int _X = <int>image.shape[ndim - 1]
        cdef int _Y = <int>image.shape[ndim - 2]
        cdef int repeats = image.size / _X / _Y
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

        num_threads = repeats if <int>num_threads > repeats else <int>num_threads

        with nogil, parallel(num_threads=num_threads):
            ldims = <unsigned long *>malloc(2 * sizeof(unsigned long))
            ldims[1] = LINE_SIZE

            for i in prange(repeats, schedule='guided'):
                fail |= LineSegmentDetection(&_outs[i], &_ns[i], _img + i * _Y * _X, _Y, _X,
                                            self.scale, self.sigma_scale, self.quant,
                                            self.ang_th, self.log_eps, self.density_th, N_BINS,
                                            &_regs[i], &_reg_ys[i], &_reg_xs[i])

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

    def draw_lines(self, np.ndarray mask not None, dict lines not None, np.ndarray idxs,
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
        cdef float **_lines = <float **>malloc(N * sizeof(float *))
        cdef unsigned long **_ldims = <unsigned long **>malloc(N * sizeof(unsigned long *))
        cdef np.ndarray _larr
        for i in range(N):
            _larr = lines[idxs[i]]
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