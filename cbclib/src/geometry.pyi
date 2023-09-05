from typing import List, Optional, Tuple, Union
import numpy as np

FloatArray = Union[List[float], Tuple[float]]

def euler_angles(rot_mats: np.ndarray, num_threads: int=1) -> np.ndarray:
    r"""Calculate Euler angles with Bunge convention [EUL]_.

    Args:
        rot_mats : A set of rotation matrices.
        num_threads : Number of threads used in the calculations.

    Returns:
        A set of Euler angles with Bunge convention :math:`\phi_1, \Phi, \phi_2`.

    References:
        .. [EUL] Depriester, Dorian. (2018), "Computing Euler angles with Bunge convention from
                rotation matrix", 10.13140/RG.2.2.34498.48321/5.
    """
    ...

def euler_matrix(angles: np.ndarray, num_threads: int=1) -> np.ndarray:
    r"""Calculate rotation matrices from Euler angles with Bunge convention [EUL]_.

    Args:
        angles : Euler angles :math:`\phi_1, \Phi, \phi_2`.
        num_threads : Number of threads used in the calculations.

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

def det_to_k(x: np.ndarray, y: np.ndarray, src: np.ndarray, idxs: Optional[np.ndarray]=None,
             num_threads: int=1) -> np.ndarray:
    """Convert coordinates on the detector ``x`, ``y`` to wave-vectors originating from
    the source points ``src``.

    Args:
        x : x coordinates in pixels.
        y : y coordinates in pixels.
        src : Source points in meters (relative to the detector).
        num_threads : Number of threads used in the calculations.

    Returns:
        A set of wave-vectors.
    """
    ...

def k_to_det(karr: np.ndarray, src: np.ndarray, idxs: Optional[np.ndarray]=None,
             num_threads: int=1) -> Tuple[np.ndarray, np.ndarray]:
    """Convert wave-vectors originating from the source points ``src`` to coordinates on
    the detector.

    Args:
        karr : An array of wave-vectors.
        src : Source points in meters (relative to the detector).
        idxs : Source point indices.
        num_threads : Number of threads used in the calculations.

    Returns:
        A tuple of x and y coordinates in meters.
    """
    ...

def k_to_smp(karr: np.ndarray, z: np.ndarray, src: FloatArray,
             idxs: Optional[np.ndarray]=None, num_threads: int=1) -> np.ndarray:
    """Convert wave-vectors originating from the source point ``src`` to sample
    planes at the z coordinate ``z``.

    Args:
        karr : An array of wave-vectors.
        src : Source point in meters (relative to the detector).
        z : Plane z coordinates in meters (relative to the detector).
        idxs : Plane indices.
        num_threads : Number of threads used in the calculations.

    Returns:
        An array of points belonging to the ``z`` planes.
    """
    ...

def rotate(vecs: np.ndarray, rmats: np.ndarray, idxs: Optional[np.ndarray]=None,
           num_threads: int=1) -> np.ndarray:
    """Rotate vectors ``vecs`` by rotation matrices ``rmats``.

    Args:
        vecs : Array of vectors.
        rmats : Array of rotation matrices.
        idxs : Rotation matrix indices.
        num_threads : Number of threads used in the calculations.

    Returns:
        An array of rotated vectors.
    """
    ...

def source_lines(hkl: np.ndarray, basis: np.ndarray, kmin: FloatArray, kmax: FloatArray,
                 num_threads: int=1) -> np.ndarray:
    r"""Calculate the source lines for a set of diffraction orders ``hkl`` and the given indexing
    solution ``basis``.

    Args:
        basis : Basis vectors of the indexing solution.
        hkl : HKL indices of diffraction orders.
        kin_min : Lower bound of the rectangular aperture function.
        kin_max : Upper bound of the rectangular aperture function.
        num_threads : A number of threads used in the computations.

    Returns:
        A set of source lines in the aperture function.
    """
    ...
