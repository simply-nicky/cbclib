from typing import Optional
from ..annotations import NDRealArray, NDIntArray, Shape, Table

def draw_line(out: NDRealArray, lines: NDRealArray, idxs: Optional[NDIntArray]=None,
              max_val: float=1.0, kernel: str='biweight', num_threads: int=1) -> NDRealArray:
    """Draw thick lines on top of a n-dimensional array `out` with variable thickness and the
    antialiasing applied on a single frame.

    Args:
        out : An output n-dimensional array.
        lines : A dictionary of the detected lines. Each array of lines must have a shape of
            (`N`, 5), where `N` is the number of lines. Each line is comprised of 5 parameters
            as follows:

            * `[x0, y0]`, `[x1, y1]` : The coordinates of the line's ends.
            * `width` : Line's width.

        shape : Shape of the output array. All the lines outside the shape will be discarded.
        idxs : An array of indices that specify to what frame each of the lines belong.
        max_val : Maximum pixel value of a drawn line.
        kernel : Choose one of the supported kernel functions [Krn]_. The following kernels
            are available:

            * 'biweigth' : Quartic (biweight) kernel.
            * 'gaussian' : Gaussian kernel.
            * 'parabolic' : Epanechnikov (parabolic) kernel.
            * 'rectangular' : Uniform (rectangular) kernel.
            * 'triangular' : Triangular kernel.

        num_threads : Number of threads used in the calculations.

    Raises:
        ValueError : If any of `lines` dictionary values have an incompatible shape.

    Returns:
        Output array with the lines drawn.
    """
    ...

def draw_line_vjp(cotangents: NDRealArray, lines: NDRealArray,
                  idxs: Optional[NDIntArray]=None, max_val: float=1.0,
                  kernel: str='biweight', num_threads: int=1) -> NDRealArray:
    ...

def draw_line_mask(lines: NDRealArray, shape: Shape, idxs: Optional[NDIntArray]=None,
                   max_val: int=255, kernel: str='rectangular',
                   num_threads: int=1) -> NDIntArray:
    """Draw thick lines with variable thickness and the antialiasing applied on a single frame
    by using the Bresenham's algorithm [BSH]_.

    Args:
        lines : A dictionary of the detected lines. Each array of lines must have a shape of
            (`N`, 5), where `N` is the number of lines. Each line is comprised of 5 parameters
            as follows:

            * `[x0, y0]`, `[x1, y1]` : The coordinates of the line's ends.
            * `width` : Line's width.

        shape : Shape of the output array. All the lines outside the shape will be discarded.
        idxs : An array of indices that specify to what frame each of the lines belong.
        max_val : Maximum pixel value of a drawn line.
        kernel : Choose one of the supported kernel functions [Krn]_. The following kernels
            are available:

            * 'biweigth' : Quartic (biweight) kernel.
            * 'gaussian' : Gaussian kernel.
            * 'parabolic' : Epanechnikov (parabolic) kernel.
            * 'rectangular' : Uniform (rectangular) kernel.
            * 'triangular' : Triangular kernel.

        num_threads : Number of threads used in the calculations.

    Raises:
        ValueError : If `lines` has an incompatible shape.

    References:
        .. [BSH] "Bresenham's line algorithm." Wikipedia, Wikimedia Foundation, 20 Sept. 2022,
                https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm.

    Returns:
        Output array with the lines drawn.
    """
    ...

def draw_line_image(lines: NDRealArray, shape: Shape, idxs: Optional[NDIntArray]=None,
                    max_val: float=1.0, kernel: str='rectangular',
                    num_threads: int=1) -> NDRealArray:
    """Draw thick lines with variable thickness and the antialiasing applied on a single frame.

    Args:
        lines : A dictionary of the detected lines. Each array of lines must have a shape of
            (`N`, 5), where `N` is the number of lines. Each line is comprised of 5 parameters
            as follows:

            * `[x0, y0]`, `[x1, y1]` : The coordinates of the line's ends.
            * `width` : Line's width.

        shape : Shape of the output array. All the lines outside the shape will be discarded.
        idxs : An array of indices that specify to what frame each of the lines belong.
        max_val : Maximum pixel value of a drawn line.
        kernel : Choose one of the supported kernel functions [Krn]_. The following kernels
            are available:

            * 'biweigth' : Quartic (biweight) kernel.
            * 'gaussian' : Gaussian kernel.
            * 'parabolic' : Epanechnikov (parabolic) kernel.
            * 'rectangular' : Uniform (rectangular) kernel.
            * 'triangular' : Triangular kernel.

        num_threads : Number of threads used in the calculations.

    Raises:
        ValueError : If any of `lines` dictionary values have an incompatible shape.

    Returns:
        Output array with the lines drawn.
    """
    ...

def draw_line_table(lines: NDRealArray, shape: Shape, idxs: Optional[NDIntArray]=None,
                    max_val: float=1.0, kernel: str='rectangular') -> Table:
    """Return an array of rasterized thick lines indices and their corresponding pixel values.
    The lines are drawn with variable thickness and the antialiasing applied.

    Args:
        lines : A dictionary of the detected lines. Each array of lines must have a shape of
            (`N`, 5), where `N` is the number of lines. Each line is comprised of 5 parameters
            as follows:

            * `[x0, y0]`, `[x1, y1]` : The coordinates of the line's ends.
            * `width` : Line's width.

        shape : Shape of the image. All the lines outside the shape will be discarded.
        idxs : An array of indices that specify to what frame each of the lines belong.
        max_val : Maximum pixel value of a drawn line.
        kernel : Choose one of the supported kernel functions [Krn]_. The following kernels
            are available:

            * 'biweigth' : Quartic (biweight) kernel.
            * 'gaussian' : Gaussian kernel.
            * 'parabolic' : Epanechnikov (parabolic) kernel.
            * 'rectangular' : Uniform (rectangular) kernel.
            * 'triangular' : Triangular kernel.

    Raises:
        ValueError : If `lines` has an incompatible shape.
        RuntimeError : If C backend exited with error.

    Returns:
        Output line indices.
    """
    ...
