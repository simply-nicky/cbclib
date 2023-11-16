from typing import Optional, Tuple
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

def kr_predict(y: np.ndarray, x: np.ndarray, x_hat: np.ndarray, sigma: float,
               kernel: str='gaussian', w: Optional[np.ndarray]=None, num_threads: int=1) -> np.ndarray:
    """Perform the multi-dimensional Nadaraya-Watson kernel regression [KerReg]_.

    Args:
        y : The data to fit.
        x : Coordinates array.
        x_hat : Set of coordinates where the fit is to be calculated.
        sigma : Kernel bandwidth.
        w : A set of weights, unitary weights are assumed if it's not provided.
        num_threads : Number of threads used in the calculations.

    Returns:
        The regression result.

    Raises:
        ValueError : If ``x`` and ``x_hat`` have incompatible shapes.
        ValueError : If ``x`` and ``y`` have incompatible shapes.

    References:
        .. [KerReg] E. A. Nadaraya, “On estimating regression,” Theory Probab. & Its Appl. 9,
            141-142 (1964).
    """
    ...
