import numpy as np

def euler_angles(rot_mats: np.ndarray) -> np.ndarray:
    r"""Calculate Euler angles with Bunge convention [EUL]_.

    Args:
        rot_mats : A set of rotation matrices.

    Returns:
        A set of Euler angles with Bunge convention :math:`\phi_1, \Phi, \phi_2`.

    References:
        .. [EUL] Depriester, Dorian. (2018), "Computing Euler angles with Bunge convention from
                rotation matrix", 10.13140/RG.2.2.34498.48321/5.
    """
    ...

def euler_matrix(angles: np.ndarray) -> np.ndarray:
    r"""Calculate rotation matrices from Euler angles with Bunge convention [EUL]_.

    Args:
        angles : Euler angles :math:`\phi_1, \Phi, \phi_2`.

    Returns:
        A set of rotation matrices.
    """
    ...

def tilt_angles(rot_mats: np.ndarray) -> np.ndarray:
    r"""Calculate an axis of rotation and a rotation angle for a rotation matrix.

    Args:
        rot_mats : A set of rotation matrices.

    Returns:
        A set of three angles :math:`\theta, \alpha, \beta`, a rotation angle :math:`\theta`, an
        angle between the axis of rotation and OZ axis :math:`\alpha`, and a polar angle of the
        axis of rotation :math:`\beta`.
        """
    ...

def tilt_matrix(angles: np.ndarray) -> np.ndarray:
    r"""Calculate a rotation matrix for a set of three angles set of three angles :math:`\theta,
    \alpha, \beta`, a rotation angle :math:`\theta`, an angle between the axis of rotation and
    OZ axis :math:`\alpha`, and a polar angle of the axis of rotation :math:`\beta`.

    Args:
        angles : A set of angles :math:`\theta, \alpha, \beta`.

    Returns:
        A set of rotation matrices.
    """
    ...

def find_rotations(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    r"""Find a rotation matrix, that transforms a vector ``a`` to the vector ``b``.

    Args:
        a : A set of first vectors.
        b : A set of second vectors.

    Returns:
        A set of rotation matrices.
    """
    ...

def cartesian_to_spherical(vecs: np.ndarray) -> np.ndarray:
    r"""Transform vectors ``vecs`` from cartesian to spherical representation.

    Args:
        vecs : A set of vectors in the cartesian coordinate system.

    Returns:
        Vectors in the spherical coordinate system.
    """
    ...

def spherical_to_cartesian(vecs: np.ndarray) -> np.ndarray:
    r"""Transform vectors ``vecs`` from spherical to cartesian representation.

    Args:
        vecs : A set of vectors in the spherical coordinate system.

    Returns:
        Vectors in the cartesian coordinate system.
    """
    ...

def gaussian_grid(x_arr: np.ndarray, y_arr: np.ndarray, z_arr: np.ndarray, basis: np.ndarray,
                  Sigma: float, sigma: float, threads: int=1) -> np.ndarray:
    r"""Criterion function for Fourier autoindexing based on maximising the marginal log-
    likelihood.

    Args:
        x_arr : A set of x coordinates of diffraction signal in Fourier space.
        y_arr : A set of y coordinates of diffraction signal in Fourier space.
        z_arr : A set of z coordinates of diffraction signal in Fourier space.
        basis : Basis vectors of the indexing solution.
        envelope : A width of the envelope gaussian function.
        sigma : A width of diffraction orders.
        threads : A number of threads used in the computations.

    Returns:
        Marginal log-likelihood.
    """
    ...

def gaussian_grid_grad(x_arr: np.ndarray, y_arr: np.ndarray, z_arr: np.ndarray, basis: np.ndarray,
                       hkl: np.ndarray, envelope: float, sigma: float, threads: int=1) -> np.ndarray:
    r"""Return the  gradient of the criterion function for Fourier autoindexing based on
    maximising the marginal log-likelihood.

    Args:
        x_arr : A set of x coordinates of diffraction signal in Fourier space.
        y_arr : A set of y coordinates of diffraction signal in Fourier space.
        z_arr : A set of z coordinates of diffraction signal in Fourier space.
        basis : Basis vectors of the indexing solution.
        envelope : A width of the envelope gaussian function.
        sigma : A width of diffraction orders.
        threads : A number of threads used in the computations.

    Returns:
        Gradient of the marginal log-likelihood.
    """
    ...

def calc_source_lines(basis: np.ndarray, hkl: np.ndarray, kin_min: np.ndarray, kin_max: np.ndarray,
                      threads: int=1) -> np.ndarray:
    r"""Calculate the source lines for a set of diffraction orders ``hkl`` and the given indexing
    solution ``basis``.

    Args:
        basis : Basis vectors of the indexing solution.
        hkl : HKL indices of diffraction orders.
        kin_min : Lower bound of the rectangular aperture function.
        kin_max : Upper bound of the rectangular aperture function.
        threads : A number of threads used in the computations.

    Returns:
        A set of source lines in the aperture function.
    """
    ...

def cross_entropy(x: np.ndarray, p: np.ndarray, q: np.ndarray, q_max: float, epsilon: float) -> float:
    """Calculate the cross-entropy criterion between an experimental pattern ``p`` and a simulated
    pattern ``q``.

    Args:
        x : A set of detector indices for the experimental pattern.
        p : Likelihood values for the experimental pattern.
        q : Simulated pattern, has the same size as the detector grid.
        q_max : Maximum value of the simulated pattern.
        epsilon : Epsilon value in the log term.

    Returns:
        Cross-entropy value.
    """
    ...

def filter_hkl(sgn: np.ndarray, bgd: np.ndarray, coord: np.ndarray, prob: np.ndarray, idxs: np.ndarray,
               threshold: float, threads: int=1) -> np.ndarray:
    """Filter generated diffraction streaks that have the signal-to-noise ratio above ``threshold``.
    The SNR value is calculated as the ratio between the absolute value of background corrected signal
    ``sgn`` and the square root of the background signal ``bgd``.

    Args:
        sgn : Background corrected measured intensities.
        bgd : Background intensities.
        coord : Coordinates of the generated pattern.
        prob : Normalised intensities of the generated pattern.
        idxs : Streak indices of the generated pattern.
        threshold : SNR ratio threshold.
        threads : Number of threads used in the calculations.

    Returns:
        A mask of diffraction streaks, True if :code:`SNR > threshold` for the given streak.
    """
    ...
