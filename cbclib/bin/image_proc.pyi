from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

def next_fast_len(target: int, backend: str='numpy') -> int:
    r"""Find the next fast size of input data to fft, for zero-padding, etc. FFT algorithms
    gain their speed by a recursive divide and conquer strategy. This relies on efficient
    functions for small prime factors of the input length. Thus, the transforms are fastest
    when using composites of the prime factors handled by the fft implementation. If there
    are efficient functions for all radices <= n, then the result will be a number x >= target
    with only prime factors < n. (Also known as n-smooth numbers)

    Args:
        target : Length to start searching from. Must be a positive integer.
        backend : Find n-smooth number for the FFT implementation from the numpy ('numpy') or
            FFTW ('fftw') library.

    Raises:
        ValueError : If `backend` is invalid.
        ValueError : If `target` is negative.

    Returns:
        The smallest fast length greater than or equal to `target`.
    """
    ...

def fft_convolve(array: np.ndarray, kernel: np.ndarray, axis: int=-1,
                 mode: str='constant', cval: float=0.0, backend: str='numpy',
                 num_threads: int=1) -> np.ndarray:
    """Convolve a multi-dimensional `array` with one-dimensional `kernel` along the
    `axis` by means of FFT. Output has the same size as `array`.

    Args:
        array : Input array.
        kernel : Kernel array.
        axis : Array axis along which convolution is performed.
        mode : The mode parameter determines how the input array is extended
            when the filter overlaps a border. Default value is 'constant'. The
            valid values and their behavior is as follows:

            * `constant`, (k k k k | a b c d | k k k k) : The input is extended by
              filling all values beyond the edge with the same constant value, defined
              by the `cval` parameter.
            * `nearest`, (a a a a | a b c d | d d d d) : The input is extended by
              replicating the last pixel.
            * `mirror`, (c d c b | a b c d | c b a b) : The input is extended by
              reflecting about the center of the last pixel. This mode is also sometimes
              referred to as whole-sample symmetric.
            * `reflect`, (d c b a | a b c d | d c b a) : The input is extended by
              reflecting about the edge of the last pixel. This mode is also sometimes
              referred to as half-sample symmetric.
            * `wrap`, (a b c d | a b c d | a b c d) : The input is extended by wrapping
              around to the opposite edge.

        cval :  Value to fill past edges of input if mode is 'constant'. Default
            is 0.0.
        backend : Choose between numpy ('numpy') or FFTW ('fftw') library for the FFT
            implementation.
        num_threads : Number of threads used in the calculations.

    Raises:
        ValueError : If `backend` is invalid.
        RuntimeError : If C backend exited with error.

    Returns:
        A multi-dimensional array containing the discrete linear convolution of `array`
        with `kernel`.
    """
    ...

def gaussian_kernel(sigma: float, order: int=0, truncate: float=4.0) -> np.ndarray:
    """Discrete Gaussian kernel.

    Args:
        sigma : Standard deviation for Gaussian kernel.
        order : The order of the filter. An order of 0 corresponds to convolution with
            a Gaussian kernel. A positive order corresponds to convolution with that
            derivative of a Gaussian. Default is 0.
        truncate : Truncate the filter at this many standard deviations. Default is 4.0.

    Returns:
        Gaussian kernel.
    """
    ...

def gaussian_filter(inp: np.ndarray, sigma: Union[float, List[float]], order: Union[int, List[int]]=0,
                    mode: str='reflect', cval: float=0.0, truncate: float=4.0, backend: str='numpy',
                    num_threads: int=1) -> np.ndarray:
    r"""Multidimensional Gaussian filter. The multidimensional filter is implemented as
    a sequence of 1-D FFT convolutions.

    Args:
        inp : The input array.
        sigma : Standard deviation for Gaussian kernel. The standard
            deviations of the Gaussian filter are given for each axis as a sequence, or as a
            single number, in which case it is equal for all axes.
        order : The order of the filter along each axis is given as a
            sequence of integers, or as a single number. An order of 0 corresponds to convolution
            with a Gaussian kernel. A positive order corresponds to convolution with that
            derivative of a Gaussian.
        mode : The mode parameter determines how the input array is extended when the
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

        cval : Value to fill past edges of input if mode is 'constant'. Default is 0.0.
        truncate : Truncate the filter at this many standard deviations. Default is 4.0.
        backend : Choose between numpy ('numpy') or FFTW ('fftw') backend library for the
            FFT implementation.
        num_threads : Number of threads.

    Raises:
        ValueError : If `backend` is invalid.
        RuntimeError : If C backend exited with error.

    Returns:
        Returned array of the same shape as `inp`.
    """
    ...

def gaussian_gradient_magnitude(inp: np.ndarray, sigma: Union[float, List[float]], mode: str='reflect',
                                cval: float=0.0, truncate: float=4.0, backend: str='numpy',
                                num_threads: int=1) -> np.ndarray:
    r"""Multidimensional gradient magnitude using Gaussian derivatives. The multidimensional
    filter is implemented as a sequence of 1-D FFT convolutions.

    Args:
        inp : The input array.
        sigma : Standard deviation for Gaussian kernel. The standard
            deviations of the Gaussian filter are given for each axis as a sequence, or as a
            single number, in which case it is equal for all axes.
        mode : The mode parameter determines how the input array is extended when the
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

        cval : Value to fill past edges of input if mode is `constant`. Default is 0.0.
        truncate : Truncate the filter at this many standard deviations. Default is 4.0.
        backend : Choose between numpy ('numpy') or FFTW ('fftw') backend library for the
            FFT implementation.
        num_threads : Number of threads.

    Raises:
        ValueError : If `backend` is invalid.
        RuntimeError : If C backend exited with error.

    Returns:
        Gaussian gradient magnitude array. The array is the same shape as `inp`.
    """
    ...

def median(inp: np.ndarray, mask: Optional[np.ndarray]=None, axis: int=0, num_threads: int=1) -> np.ndarray:
    """Calculate a median along the `axis`.

    Args:
        inp : Input array. Must be one of the following types: np.float64, np.float32, np.int32,
            np.uint32, np.uint64.
        mask : Output mask. Median is calculated only where `mask` is True, output array set to 0
            otherwise. Median is calculated over the whole input array by default.
        axis : Array axis along which median values are calculated.
        num_threads : Number of threads used in the calculations.

    Raises:
        ValueError : If `mask` and `inp` have different shapes.
        TypeError : If `inp` has incompatible type.
        RuntimeError : If C backend exited with error.

    Returns:
        Array of medians along the given axis.
    """
    ...

def median_filter(inp: np.ndarray, size: Optional[Union[int, Tuple[int, ...]]]=None,
                  footprint: Optional[np.ndarray]=None, mask: Optional[np.ndarray]=None,
                  inp_mask: Optional[np.ndarray]=None, mode: str='reflect', cval: float=0.0,
                  num_threads: int=1) -> np.ndarray:
    """Calculate a multidimensional median filter.

    Args:
        inp : Input array. Must be one of the following types: np.float64, np.float32, np.int32,
            np.uint32, np.uint64.
        size : See footprint, below. Ignored if footprint is given.
        footprint :  Either size or footprint must be defined. size gives the shape that is taken
            from the input array, at every element position, to define the input to the filter
            function. footprint is a boolean array that specifies (implicitly) a shape, but also
            which of the elements within this shape will get passed to the filter function. Thus
            size=(n, m) is equivalent to footprint=np.ones((n, m)). We adjust size to the number of
            dimensions of the input array, so that, if the input array is shape (10, 10, 10), and
            size is 2, then the actual size used is (2, 2, 2). When footprint is given, size is
            ignored.
        mask : Output mask. Median is calculated only where `mask` is True, output array set to 0
            otherwise. Median is calculated over the whole input array by default.
        inp_mask : Input mask. Median takes into account only the `inp` values, where `inp_mask`
            is True. `inp_mask` is equal to `mask` by default.
        mode : The mode parameter determines how the input array is extended when the
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
        cval : Value to fill past edges of input if mode is 'constant'. Default is 0.0.
        num_threads : Number of threads used in the calculations.

    Raises:
        ValueError : When neither `size` nor `footprint` are provided.
        TypeError : If `data` has incompatible type.
        RuntimeError : If C backend exited with error.

    Returns:
        Filtered array. Has the same shape as `inp`.
    """
    ...

def maximum_filter(inp: np.ndarray, size: Optional[Union[int, Tuple[int, ...]]],
                   footprint: Optional[np.ndarray]=None, mask: Optional[np.ndarray]=None,
                   mode: str='reflect', cval: float=0.0, num_threads: int=1) -> np.ndarray:
    """Calculate a multidimensional maximum filter.

    Args:
        inp : Input array. Must be one of the following types: np.float64, np.float32, np.int32,
            np.uint32, np.uint64.
        size: See footprint, below. Ignored if footprint is given.
        footprint :  Either size or footprint must be defined. size gives the shape that is taken
            from the input array, at every element position, to define the input to the filter
            function. footprint is a boolean array that specifies (implicitly) a shape, but also
            which of the elements within this shape will get passed to the filter function. Thus
            size=(n, m) is equivalent to footprint=np.ones((n, m)). We adjust size to the number of
            dimensions of the input array, so that, if the input array is shape (10, 10, 10), and
            size is 2, then the actual size used is (2, 2, 2). When footprint is given, size is
            ignored.
        mask : Output mask. Median is calculated only where `mask` is True, output array set to 0
            otherwise. Median is calculated over the whole input array by default.
        mode : The mode parameter determines how the input array is extended when the
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
        cval : Value to fill past edges of input if mode is 'constant'. Default is 0.0.
        num_threads : Number of threads.

    Raises:
        ValueError : When neither `size` nor `footprint` are provided.
        TypeError : If `data` has incompatible type.
        RuntimeError : If C backend exited with error.

    Returns:
        Filtered array. Has the same shape as `inp`.
    """
    ...

def draw_line_mask(shape: Tuple[int, ...], lines: Union[np.ndarray, Sequence[np.ndarray]], max_val: int=255,
                   dilation: float=0.0, profile: str='tophat') -> np.ndarray:
    """Draw thick lines with variable thickness and the antialiasing applied on a single frame
    by using the Bresenham's algorithm [BSH]_. The lines must follow the LSD convention,
    see the parameters for more info.

    Args:
        inp : Input array.
        lines : An array of the detected lines. Must have a shape of (`N`, 7), where `N` is
            the number of lines. Each line is comprised of 7 parameters as follows:

            * `[x0, y0]`, `[x1, y1]` : The coordinates of the line's ends.
            * `width` : Line's width.
            * `p` : Angle precision [0, 1] given by angle tolerance over 180 degree.
            * `-log10(NFA)` : Number of false alarms.

        max_val : Maximum pixel value of a drawn line.
        dilation : Size of the binary dilation applied to the output array.
        profile : Line width profiles. The following keyword values are allowed:

            * `tophat` : Top-hat (rectangular) function profile.
            * `linear` : Linear (triangular) function profile.
            * `quad` : Quadratic (parabola) function profile.
            * `gauss` : Gaussian funtion profile.

    Raises:
        ValueError : If `inp` is not a 2-dimensional array.
        ValueError : If `lines` has an incompatible shape.
        RuntimeError : If C backend exited with error.

    References:
        .. [BSH] "Bresenham's line algorithm." Wikipedia, Wikimedia Foundation, 20 Sept. 2022,
                https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm.

    Returns:
        Output array with the lines drawn.

    See Also:
        :class:`cbclib.bin.LSD` : Line Segment Detector.
    """
    ...

def draw_line_image(shape: Tuple[int, ...], lines: Union[np.ndarray, Sequence[np.ndarray]],
                    dilation: float=0.0, profile: str='gauss', num_threads: int=1) -> np.ndarray:
    """Draw thick lines with variable thickness and the antialiasing applied on a single frame.
    The lines must follow the LSD convention, see the parameters for more info.

    Args:
        inp : Input array.
        lines : A dictionary of the detected lines. Each array of lines must have a shape of
            (`N`, 7), where `N` is the number of lines. Each line is comprised of 7 parameters
            as follows:

            * `[x0, y0]`, `[x1, y1]` : The coordinates of the line's ends.
            * `width` : Line's width.
            * `p` : Angle precision [0, 1] given by angle tolerance over 180 degree.
            * `-log10(NFA)` : Number of false alarms.

        max_val : Maximum pixel value of a drawn line.
        dilation : Size of the binary dilation applied to the output array.
        profile : Line width profiles. The following keyword values are allowed:

            * `tophat` : Top-hat (rectangular) function profile.
            * `linear` : Linear (triangular) function profile.
            * `quad` : Quadratic (parabola) function profile.
            * `gauss` : Gaussian funtion profile.

        num_threads : Number of threads used in the calculations.

    Raises:
        ValueError : If `inp` number of dimensions is less than 3.
        ValueError : If any of `lines` dictionary values have an incompatible shape.
        RuntimeError : If C backend exited with error.

    Returns:
        Output array with the lines drawn.

    See Also:
        :class:`cbclib.bin.LSD` : Line Segment Detector.
    """
    ...

def draw_line_table(lines: np.ndarray, shape: Optional[Tuple[int, int]]=None, dilation: float=0.0,
                    profile: str='gauss') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return an array of rasterized thick lines indices and their corresponding pixel values.
    The lines are drawn with variable thickness and the antialiasing applied. The lines must
    follow the LSD convention, see the parameters for more info.

    Args:
        lines : An array of the detected lines. Must have a shape
            of (`N`, 7), where `N` is the number of lines. Each line is comprised
            of 7 parameters as follows:

            * `[x0, y0]`, `[x1, y1]` : The coordinates of the line's ends.
            * `width` : Line's width.
            * `p` : Angle precision [0, 1] given by angle tolerance over 180 degree.
            * `-log10(NFA)` : Number of false alarms.

        shape : Shape of the image. All the lines outside the shape will be discarded.
        max_val : Maximum pixel value of a drawn line.
        dilation : Size of the binary dilation applied to the output image.
        profile : Line width profiles. The following keyword values are allowed:

            * `tophat` : Top-hat (rectangular) function profile.
            * `linear` : Linear (triangular) function profile.
            * `quad` : Quadratic (parabola) function profile.
            * `gauss` : Gaussian funtion profile.

    Raises:
        ValueError : If `lines` has an incompatible shape.
        RuntimeError : If C backend exited with error.

    Returns:
        Output line indices.

    See Also:
        :class:`cbclib.bin.LSD` : Line Segment Detector.
    """
    ...

def outlier_rate(data: np.ndarray, bgd: np.ndarray, iidxs: np.ndarray, hkl_idxs: np.ndarray,
                 alpha: float, num_threads: int=1) -> Tuple[np.ndarray, np.ndarray]:
    r"""Count the outliers for a set of diffraction orders, which photon counts are above the
    ``alpha`` Poisson distribution quantile with expected values equal to the background
    intensities ``bgd``.

    Args:
        data : Photon counts.
        bgd : Background intensities.
        iidxs : Streak indices of the generated pattern.
        hkl_idxs : Diffraction order indices.
        alpha : Quantile level, which must be between 0 and 1 inclusive.
        num_threads : Number of threads used in the calculations.

    Notes:
        The confidence interval for the mean of a Poisson distribution can be expressed using
        the relationship between the cumulative distribution functions of the Poisson and
        chi-squared distributions. The chi-squared distribution is itself closely related to
        the gamma distribution, and this leads to an alternative expression. Given an observation
        ``k`` from a Poisson distribution with mean :math:`\mu`, a confidence interval for
        :math:`\mu` with confidence level :math:`1 - \alpha` is:

        .. math::
            \frac{1}{2} \chi^2(\alpha / 2, 2 k) \leq \mu \leq \frac{1}{2}
            \chi^2(1 - \alpha / 2, 2 k + 2),

        where :math:`\chi^2(p, n)` is the quantile function of the chi-squared distribuion with n
        degrees.

    Returns:
        An array of outlier and total counts.
    """
    ...

def normalise_pattern(inp: np.ndarray, lines: Dict[int, np.ndarray], dilations: Tuple[float, float, float],
                      profile: str='tophat', num_threads: int=1) -> np.ndarray:
    """Perform the normalisation of measured CBC patterns ``inp`` based on two-zone masking.
    The inner and outer zone are defined by a set of dilation radii ``dilations``. The normalised
    reflection is given by :math:`(inp[i, j] - b) / (c - b)`, where ``c`` is the maximum intensity
    in the inner zone and ``b`` is the median intensity in the outer zone.

    Args:
        inp : Measured CBC patterns.
        lines : Detected diffraction streaks.
        dilations : Dilation radii of two-zone masking in pixels.
        profile : Line width profile used to calculate ``c`` and ``b``. The following keyword
            values are allowed:

            * `tophat` : Top-hat (rectangular) function profile.
            * `linear` : Linear (triangular) function profile.
            * `quad` : Quadratic (parabola) function profile.
            * `gauss` : Gaussian funtion profile.

        num_threads : Number of threads used in the calculations.

    Returns:
        Normalised CBC patterns.
    """
    ...

def refine_pattern(inp: np.ndarray, lines: Dict[int, np.ndarray], dilation: float, profile: str='tophat',
                   num_threads: int=1) -> Dict[int, np.ndarray]:
    """Refine detected diffraction streaks by fitting a Gaussian across the line.

    Args:
        inp : Measured CBC patterns.
        lines : Detected diffraction streaks.
        dilation : Dilation radius in pixels used for the Gaussian fit.
        profile : Line width profile used for the Gaussian fit. The following keyword values
            are allowed:

            * `tophat` : Top-hat (rectangular) function profile.
            * `linear` : Linear (triangular) function profile.
            * `quad` : Quadratic (parabola) function profile.
            * `gauss` : Gaussian funtion profile.            

        num_threads : Number of threads used in the calculations.

    Returns:
        Refined diffraction streaks.
    """
    ...

def project_effs(inp: np.ndarray, mask: np.ndarray, effs: np.ndarray, num_threads: int=1) -> np.ndarray:
    """Calculate a projection of eigen flat-fields ``effs`` on a set of 2D arrays ``inp``.

    Args:
        inp : A set of 2D arrays.
        mask : A set of 2D masks.
        effs : A set of eigen flat-fields.
        num_threads : A number of threads used in the computations.

    Returns:
        An output projection of eigen flat-fields.
    """
    ...

def subtract_background(inp: np.ndarray, mask: np.ndarray, bgd: np.ndarray, num_threads: int=1) -> np.ndarray:
    """Subtract background from a set of 2D arrays ``inp``.

    Args:
        inp : A set of 2D arrays.
        mask : A set of 2D masks.
        bgd : A set of background arrays.
        num_threads : A number of threads used in the computations.

    Returns:
        An output projection of eigen flat-fields.
    """
    ...

def ce_criterion(ij: np.ndarray, p: np.ndarray, fidxs: np.ndarray, shape: Tuple[int, int],
                 lines: Sequence[np.ndarray], dilation: float=0.0, epsilon: float=1e-12,
                 profile: str='gauss', num_threads: int=1) -> float:
    r"""Calculate the cross-entropy criterion between an experimental pattern ``p`` and
    simulated diffraction streaks ``lines``.

    Args:
        ij : Detector coordinates, where the measured pattern ``p`` is above zero.
        p : Measured normalised intensities.
        fidxs : Frame indices.
        shape : Shape of the detector pixel grid.
        lines : Simulated diffraction streaks.
        dilation : Dilation radius in pixels used to rasterise simulated diffraction
            streaks ``lines``.
        epsilon : Epsilon value used to calculated logarithm of simulated standart
            profiles.

    Notes:
        The cross-entropy between the measured patterns :math:`i_n(\mathbf{x}_i)` and the
        simulated streaks is given by:

        .. math::
            \mathcal{L} = -\sum_{ni} i_n(\mathbf{x}_i) \log(\max(f^2_{hkl},
            \epsilon)),

        where :math:`f^2_{hkl}` is a set of standard profile patterns calculated with the
        help of Bresenham's algorithm.

    Returns:
        Cross-entropy value.
    """
    ...
