from typing import Tuple
import numpy as np

def binterpolate(inp: np.ndarray, grid: Tuple[np.ndarray, ...], coords: np.ndarray, num_threads: int=1) -> np.ndarray:
    """Perform bilinear multidimensional interpolation on regular grids. The integer grid starting
    from ``(0, 0, ...)`` to ``(inp.shape[0] - 1, inp.shape[1] - 1, ...)`` is implied.

    Args:
        inp : The data on the regular grid in n dimensions.
        grid : A tuple of grid coordinates along each dimension (`x`, `y`, `z`, ...).
        coords : A list of N coordinates [(`x_hat`, `y_hat`, `z_hat`), ...] to sample the gridded
            data at.
        num_threads : Number of threads used in the calculations.

    Raises:
        ValueError : If ``inp`` and ``coords`` have incompatible shapes.

    Returns:
        Interpolated values at input coordinates.
    """
    ...
