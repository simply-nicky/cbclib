from typing import Optional, Tuple
import numpy as np

def unique_indices(idxs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find unique ``frames``, the indices of the input array that give the unique ``frames``, and
    indices that give the unique ``idxs``.

    Args:
        frames : An array of frames.
        idxs : An array of indices.

    Returns:
        Return a tuple of three items (`uniq`, `iidxs`). The elements of the tuple are as
        follows:

        * `uniq` : Unique ``frames`` values.
        * `iidxs` : Indices of the input array that give the unique ``idxs``.
    """
    ...

def kr_predict(y: np.ndarray, x: np.ndarray, x_hat: np.ndarray, sigma: float,
               w: Optional[np.ndarray]=None, num_threads: int=1) -> np.ndarray:
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

def kr_grid(y: np.ndarray, x: np.ndarray, grid: Tuple[np.ndarray, ...], sigma: float,
            w: Optional[np.ndarray]=None, return_roi: bool=True,
            num_threads: int=1) -> Tuple[np.ndarray, np.ndarray]:
    """Perform the multi-dimensional Nadaraya-Watson kernel regression [KerReg]_ over a grid of
    points.

    Args:
        y : The data to fit.
        x : Coordinates array.
        step : Grid sampling interval.
        sigma : Kernel bandwidth.
        w : A set of weights, unitary weights are assumed if it's not provided.
        return_roi : Return region of interest of the sampling grid if True.
        num_threads : Number of threads used in the calculations.

    Raises:
        ValueError : If ``step`` is negative.

    Returns:
        The regression result and the region of interest if ``return_roi`` is True.
    """
    ...

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

def poisson_criterion(x: np.ndarray, ij: np.ndarray, shape: Tuple[int, int], I0: np.ndarray, bgd: np.ndarray,
                      xtal_bi: np.ndarray, prof: np.ndarray, fidxs: np.ndarray, idxs: np.ndarray,
                      hkl_idxs: np.ndarray, oidxs: Optional[np.ndarray]=None,
                      num_threads: int=1) -> Tuple[np.ndarray, np.ndarray]:
    r"""Calculate the Poisson negative log likelihood that the measured intensities ``I0`` are
    explained by the current estimate of crystal structure factors ``x`` and sample projection
    maps ``xtal_bi``.

    Args:
        x : Current estimate of crystal structure factors and intercept values.
        ij : Detector coordinates.
        shape : Shape of the detector grid.
        I0 : Measured diffracted signal.
        bgd : Background level.
        xtal_bi : Sample's projection maps.
        prof : Standard profiles.
        fidxs : Frame indices.
        idxs : Streak indices.
        hkl_idxs : Set of indices that numerate different Bragg reflections.
        oidxs : Output criterion indices.
        num_threads : Number of threads used in the calculations.

    Notes:
        The intensity profile :math:`I_{hkl}(\mathbf{x})` of a particular Bragg reflection
        captured on the detector is given by:

        .. math::
            I_{hkl}(\mathbf{x}) = |q_{hkl}|^2 \chi(\mathbf{u}(\mathbf{x})) f^2_{hkl}(\mathbf{x})

        where :math:`q_{hkl}` are the structure factors and :math:`\chi(\mathbf{u}(\mathbf{x}))`
        are the projection maps of the sample, and :math:`f_{hkl}(\mathbf{x})` are the standard
        reflection profiles.

        The Poisson negative log likelihood crietion is given by:

        .. math::
            \epsilon^{NLL} = \sum_{ni} \log \mathrm{P}(I_n(\mathbf{x}_i), I_{hkl}(\mathbf{x}_i)
            + I_{bgd}(\mathbf{x}_i)),

        where the likelihood :math:`\mathrm{P}` follows the Poisson distribution :math:`\log
        \mathrm{P}(I, \lambda) = I \log \lambda - I`.

    Returns:
        Negative log likelihood and gradient arrays.
    """
    ...

def ls_criterion(x: np.ndarray, ij: np.ndarray, shape: Tuple[int, int], I0: np.ndarray, bgd: np.ndarray,
                 xtal_bi: np.ndarray, prof: np.ndarray, fidxs: np.ndarray, idxs: np.ndarray,
                 hkl_idxs: np.ndarray, oidxs: Optional[np.ndarray]=None, loss: str='l2',
                 num_threads: int=1) -> Tuple[np.ndarray, np.ndarray]:
    r"""Calculate the least-squares error between the measured intensities ``I0`` and the
    modelled intenisty profiles of Bragg reflections.

    Args:
        x : Current estimate of crystal structure factors and intercept values.
        ij : Detector coordinates.
        shape : Shape of the detector grid.
        I0 : Measured diffracted signal.
        bgd : Background level.
        xtal_bi : Sample's projection maps.
        prof : Standard profiles.
        fidxs : Frame indices.
        idxs : Streak indices.
        hkl_idxs : Set of indices that numerate different Bragg reflections.
        oidxs : Output criterion indices.
        loss : Loss function used to calculate the MSE. The following keyword arguments are
            allowed:

            * `l1`: L1 loss (absolute) function.
            * `l2` : L2 loss (squared) function.
            * `Huber` : Huber loss function.

        num_threads : Number of threads used in the calculations.

    Notes:
        The intensity profile :math:`I_{hkl}(\mathbf{x})` of a particular Bragg reflection
        captured on the detector is given by:

        .. math::
            I_{hkl}(\mathbf{x}) = |q_{hkl}|^2 \chi(\mathbf{u}(\mathbf{x})) f^2_{hkl}(\mathbf{x})

        where :math:`q_{hkl}` are the structure factors and :math:`\chi(\mathbf{u}(\mathbf{x}))`
        are the projection maps of the sample, and :math:`f_{hkl}(\mathbf{x})` are the standard
        reflection profiles.

        The least squares criterion is given by:

        .. math::
            \epsilon^{LS} = \sum_{ni} f\left( \frac{I_n(\mathbf{x}_i) - I_{hkl}(\mathbf{x}_i) -
            I_{bgd}}{\sigma_I^2} \right),

        where :math:`f(x)` is either l2, l1, or Huber loss function, and :math:`\sigma_I` is the
        standard deviation of measured photon counts for a given diffraction streak.

    Returns:
        The least squares criterion and gradient arrays.
    """
    ...

def unmerge_signal(x: np.ndarray, ij: np.ndarray, shape: Tuple[int, int], I0: np.ndarray,
                   bgd: np.ndarray, xtal_bi: np.ndarray, prof: np.ndarray, fidxs: np.ndarray,
                   idxs: np.ndarray, hkl_idxs: np.ndarray, num_threads: int=1) -> np.ndarray:
    """Unmerge photon counts ``I0`` into diffraction orders ``hkl_idxs`` based on the current
    estimate of crystal structure factors and intercept values ``x``.

    Args:
        x : Current estimate of crystal structure factors and intercept values.
        prof : Standard profiles.
        I0 : Measured diffracted signal.
        bgd : Background level.
        xtal_bi : Sample's projection maps.
        hkl_idxs : Set of indices that numerate different Bragg reflections.
        iidxs : Array of first indices pertaining to different diffraction streaks.
        num_threads : Number of threads used in the calculations.

    Returns:
        An array of unmerged and background subtracted photon counts.
    """
    ...
