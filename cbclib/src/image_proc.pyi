from typing import List, Optional, Sequence, Tuple, Union
import numpy as np

def draw_line_mask(lines: Union[np.ndarray, Sequence[np.ndarray]], shape: Tuple[int, ...], max_val: int=255,
                   dilation: float=0.0, profile: str='tophat', num_threads: int=1) -> np.ndarray:
    """Draw thick lines with variable thickness and the antialiasing applied on a single frame
    by using the Bresenham's algorithm [BSH]_. The lines must follow the LSD convention,
    see the parameters for more info.

    Args:
        lines : An array of the detected lines. Must have a shape of (`N`, 7), where `N` is
            the number of lines. Each line is comprised of 7 parameters as follows:

            * `[x0, y0]`, `[x1, y1]` : The coordinates of the line's ends.
            * `width` : Line's width.
            * `p` : Angle precision [0, 1] given by angle tolerance over 180 degree.
            * `-log10(NFA)` : Number of false alarms.

        shape : Shape of the output array. All the lines outside the shape will be discarded.
        max_val : Maximum pixel value of a drawn line.
        dilation : Size of the binary dilation applied to the output array.
        profile : Line width profiles. The following keyword values are allowed:

            * `tophat` : Top-hat (rectangular) function profile.
            * `linear` : Linear (triangular) function profile.
            * `quad` : Quadratic (parabola) function profile.
            * `gauss` : Gaussian funtion profile.

        num_threads : Number of threads used in the calculations.

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

def draw_line_image(lines: Union[np.ndarray, Sequence[np.ndarray]], shape: Tuple[int, ...],
                    max_val: float=1.0, dilation: float=0.0, profile: str='gauss',
                    num_threads: int=1) -> np.ndarray:
    """Draw thick lines with variable thickness and the antialiasing applied on a single frame.
    The lines must follow the LSD convention, see the parameters for more info.

    Args:
        lines : A dictionary of the detected lines. Each array of lines must have a shape of
            (`N`, 7), where `N` is the number of lines. Each line is comprised of 7 parameters
            as follows:

            * `[x0, y0]`, `[x1, y1]` : The coordinates of the line's ends.
            * `width` : Line's width.
            * `p` : Angle precision [0, 1] given by angle tolerance over 180 degree.
            * `-log10(NFA)` : Number of false alarms.

        shape : Shape of the output array. All the lines outside the shape will be discarded.
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

def draw_line_table(lines: np.ndarray, shape: Optional[Tuple[int, int]]=None, max_val: float=1.0,
                    dilation: float=0.0, profile: str='gauss') -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
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
