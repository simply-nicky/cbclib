from typing import Optional, Tuple
import numpy as np

def unique_indices(frames: np.ndarray, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find unique ``frames``, the indices of the input array that give the unique ``frames``, and
    indices that give the unique ``indices``.

    Args:
        frames : An array of frames.
        indices : An array of indices.

    Returns:
        Return a tuple of three items (`funiq`, `fidxs`, `iidxs`). The elements of the tuple are as
        follows:

        * `funiq` : Unique ``frames`` values.
        * `fidxs` : Indices of the input array that give the unique ``frames``.
        * `iidxs` : Indices of the input array that give the unique ``indices``.
    """
    ...

def find_kins(x: np.ndarray, y: np.ndarray, hkl: np.ndarray, fidxs: np.ndarray,
              smp_pos: np.ndarray, rot_mat: np.ndarray, basis: np.ndarray,
              x_pixel_size: float, y_pixel_size: float, num_threads: int=1) -> np.ndarray:
    """Convert detector coordinates to incoming wavevectors. The incoming wavevectors are normalised
    and specify the spatial frequencies of the incoming beam that bring about the diffraction signal
    at a given coordinate on the detector.

    Args:
        x : Detector x coordinates in pixels.
        y : Detector y coordinates in pixels.
        hkl : Miller indices.
        fidxs : Indices that give the unique frames.
        smp_pos : Sample positions for each frame.
        rot_mat : Rotation matrices of the sample for each frames.
        basis : Basis vectors of crystal lattice unit cell.
        x_pixel_size : Detector pixel size along the x axis in meters.
        y_pixel_size : Detector pixel size along the y axis in meters.
        num_threads : Number of threads used in the calculations.

    Returns:
        A set of incoming wavevectors.
    """
    ...

def update_sf(bp: np.ndarray, sgn: np.ndarray, xidx: np.ndarray, xmap: np.ndarray, xtal: np.ndarray,
              hkl_idxs: np.ndarray, iidxs: np.ndarray, num_threads: int=1) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate crystal structure factors by using the ordinary least squares solution.

    Args:
        bp : Bragg profile.
        sgn : Diffraction signal.
        xidx : Crystal diffraction map frame indices.
        xmap : Mapping of diffraction signal into the crystal plane grid.
        xtal : Crystal diffraction efficiency map.
        hkl_idxs : Miller indices.
        iidxs : Indices on the input arrays that give the unique diffraction streak indices.
        num_threads : Number of threads used in the calculations.

    Raises:
        ValueError : If ``bp``, ``sgn``, ``xidx``, and ``xmap`` have incompatible shapes.
        ValueError : If ``iiidxs`` last index is not equal to ``sgn`` size.

    Returns:
        A new set of crystal structure factors and structure factor uncertainties.
    """
    ...

def scaling_criterion(sf: np.ndarray, bp: np.ndarray, sgn: np.ndarray, xidx: np.ndarray, xmap: np.ndarray,
                      xtal: np.ndarray, iidxs: np.ndarray, num_threads: int=1) -> float:
    r"""Return the mean abolute error (MAE) of the CBC dataset intensity scaling.

    Args:
        sf : Crystal structure factors.
        bp : Bragg profile
        sgn : Diffraction signal.
        xidx : Crystal diffraction map frame indices.
        xmap : Mapping of diffraction signal into the crystal plane grid.
        xtal : Crystal diffraction efficiency map.
        iidxs : Indices on the input arrays that give the unique diffraction streak indices.
        num_threads : Number of threads used in the calculations.

    Notes:
        The MAE is given by:

        .. math::
            L(I, D_{xtal}, F_{xtal}) = \frac{1}{N} \sum_{i = 0}^N \left| I - D_{xtal}(x_i, y_i)
            p_{bragg}(x_i, y_i) F_{xtal}(hkl_i) \right|,

        where :math:`I` - diffraction signal, :math:`p_{bragg}` - Bragg reflection profile,
        :math:`D_{xtal}` - crystal diffraction efficiency map, and
        :math:`F_{xtal}` - crystal structure factors.

    Raises:
        ValueError : If ``bp``, ``sf``, ``sgn``, ``xidx``, and ``xmap`` have incompatible shapes.
        ValueError : If ``iiidxs`` last index is not equal to ``sf`` size.

    Returns:
        Mean absolute error.
    """
    ...

def kr_predict(y: np.ndarray, x: np.ndarray, x_hat: np.ndarray, sigma: float, cutoff: float,
               w: Optional[np.ndarray]=None, num_threads: int=1) -> np.ndarray:
    """Perform the multi-dimensional Nadaraya-Watson kernel regression [KerReg]_.

    Args:
        y : The data to fit.
        x : Coordinates array.
        x_hat : Set of coordinates where the fit is to be calculated.
        sigma : Kernel bandwidth.
        cutoff : Distance cutoff used for the calculations.
        w : A set of weights, unitary weights are assumed if it's not provided.
        num_threads : Number of threads used in the calculations.

    Returns:
        The regression result.

    Raises:
        ValueError : If ``x`` and ``x_hat`` have incompatible shapes.
        ValueError : If ``x`` and ``y`` have incompatible shapes.

    References:
        .. [KerReg] E. A. Nadaraya, “On estimating regression,” Theory Probab. & Its
                    Appl. 9, 141-142 (1964).
    """
    ...

def kr_grid(y: np.ndarray, x: np.ndarray, step: Tuple[float, float], sigma: float,
            cutoff: float, w: Optional[np.ndarray]=None, return_roi: bool=True,
            num_threads: int=1) -> Tuple[np.ndarray, np.ndarray]:
    """Perform the multi-dimensional Nadaraya-Watson kernel regression [KerReg]_ over a grid of
    points.

    Args:
        y : The data to fit.
        x : Coordinates array.
        step : Grid sampling interval.
        sigma : Kernel bandwidth.
        cutoff : Distance cutoff used for the calculations.
        w : A set of weights, unitary weights are assumed if it's not provided.
        return_roi : Return region of interest of the sampling grid if True.
        num_threads : Number of threads used in the calculations.

    Raises:
        ValueError : If ``step`` is negative.

    Returns:
        The regression result and the region of interest if ``return_roi`` is True.
    """
    ...

def xtal_interpolate(xidx: np.ndarray, xmap: np.ndarray, xtal: np.ndarray, num_threads: int=1) -> np.ndarray:
    """Find the crystal efficiency values at the given coordinates by using the bilinear
    interpolation.

    Args:
        xidx : Crystal diffraction map frame indices.
        xmap : Mapping of diffraction signal into the crystal plane grid.
        xtal : Crystal diffraction efficiency map.

    Raises:
        ValueError : If ``xidx`` and ``xmap`` have incompatible shapes.

    Returns:
        Interpolated crystall efficiency values.
    """
    ...
