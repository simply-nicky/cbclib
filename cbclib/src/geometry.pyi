from typing import List, Optional, Tuple
from ..annotations import NDRealArray, NDIntArray, RealSequence

def euler_angles(rmats: NDRealArray, num_threads: int=1) -> NDRealArray:
    r"""Calculate Euler angles with Bunge convention [EUL]_.

    Args:
        rmats : A set of rotation matrices.
        num_threads : Number of threads used in the calculations.

    Returns:
        A set of Euler angles with Bunge convention :math:`\phi_1, \Phi, \phi_2`.

    References:
        .. [EUL] Depriester, Dorian. (2018), "Computing Euler angles with Bunge convention from
                rotation matrix", 10.13140/RG.2.2.34498.48321/5.
    """
    ...

def euler_angles_vjp(cotangents: NDRealArray, rmats: NDRealArray,
                     num_threads: int=1) -> NDRealArray:
    ...

def euler_matrix(angles: NDRealArray, num_threads: int=1) -> NDRealArray:
    r"""Calculate rotation matrices from Euler angles with Bunge convention [EUL]_.

    Args:
        angles : Euler angles :math:`\phi_1, \Phi, \phi_2`.
        num_threads : Number of threads used in the calculations.

    Returns:
        A set of rotation matrices.
    """
    ...

def euler_matrix_vjp(cotangents: NDRealArray, angles: NDRealArray,
                     num_threads: int=1) -> NDRealArray:
    ...

def tilt_angles(rmats: NDRealArray, num_threads: int=1) -> NDRealArray:
    r"""Calculate an axis of rotation and a rotation angle for a rotation matrix.

    Args:
        rmats : A set of rotation matrices.
        num_threads : Number of threads used in the calculations.

    Returns:
        A set of three angles :math:`\theta, \alpha, \beta`, a rotation angle :math:`\theta`, an
        angle between the axis of rotation and OZ axis :math:`\alpha`, and a polar angle of the
        axis of rotation :math:`\beta`.
        """
    ...

def tilt_angles_vjp(contangents: NDRealArray, rmats: NDRealArray,
                    num_threads: int=1) -> NDRealArray:
    ...

def tilt_matrix(angles: NDRealArray, num_threads: int=1) -> NDRealArray:
    r"""Calculate a rotation matrix for a set of three angles set of three angles :math:`\theta,
    \alpha, \beta`, a rotation angle :math:`\theta`, an angle between the axis of rotation and
    OZ axis :math:`\alpha`, and a polar angle of the axis of rotation :math:`\beta`.

    Args:
        angles : A set of angles :math:`\theta, \alpha, \beta`.

    Returns:
        A set of rotation matrices.
    """
    ...

def tilt_matrix_vjp(contangents: NDRealArray, angles: NDRealArray,
                    num_threads: int=1) -> NDRealArray:
    ...

def fill_indices(xsize: int, isize: int, idxs: Optional[NDIntArray]=None) -> NDIntArray:
    ...

def det_to_k(x: NDRealArray, y: NDRealArray, src: NDRealArray,
             idxs: Optional[NDIntArray]=None,
             num_threads: int=1) -> NDRealArray:
    """Convert coordinates on the detector ``x`, ``y`` to wave-vectors originating from
    the source points ``src``.

    Args:
        x : x coordinates in pixels.
        y : y coordinates in pixels.
        src : Source points in meters (relative to the detector).
        idxs : Source point indices.
        num_threads : Number of threads used in the calculations.

    Returns:
        A set of wave-vectors.
    """
    ...

def det_to_k_vjp(cotangents: NDRealArray, x: NDRealArray, y: NDRealArray,
                 src: NDRealArray, idxs: Optional[NDIntArray]=None,
                 num_threads: int=1) -> Tuple[NDRealArray, NDRealArray, NDRealArray]:
    ...

def k_to_det(k: NDRealArray, src: NDRealArray, idxs: Optional[NDIntArray]=None,
             num_threads: int=1) -> Tuple[NDRealArray, NDRealArray]:
    """Convert wave-vectors originating from the source points ``src`` to coordinates on
    the detector.

    Args:
        k : An array of wave-vectors.
        src : Source points in meters (relative to the detector).
        idxs : Source point indices.
        num_threads : Number of threads used in the calculations.

    Returns:
        A tuple of x and y coordinates in meters.
    """
    ...

def k_to_det_vjp(cotangents: Tuple[NDRealArray, NDRealArray], k: NDRealArray,
                 src: NDRealArray, idxs: Optional[NDIntArray]=None,
                 num_threads: int=1) -> Tuple[NDRealArray, NDRealArray]:
    ...

def k_to_smp(k: NDRealArray, z: NDRealArray, src: RealSequence,
             idxs: Optional[NDIntArray]=None, num_threads: int=1) -> NDRealArray:
    """Convert wave-vectors originating from the source point ``src`` to sample
    planes at the z coordinate ``z``.

    Args:
        k : An array of wave-vectors.
        src : Source point in meters (relative to the detector).
        z : Plane z coordinates in meters (relative to the detector).
        idxs : Plane indices.
        num_threads : Number of threads used in the calculations.

    Returns:
        An array of points belonging to the ``z`` planes.
    """
    ...

def k_to_smp_vjp(cotangents: NDRealArray, k: NDRealArray, z: NDRealArray,
                 src: RealSequence, idxs: Optional[NDIntArray]=None,
                 num_threads: int=1) -> Tuple[NDRealArray, NDRealArray, List[float]]:
    ...

def matmul_indices(vsize: int, msize: int, vidxs: Optional[NDIntArray]=None,
                   midxs: Optional[NDIntArray]=None) -> Tuple[NDIntArray, NDIntArray]:
    ...

def matmul(vecs: NDRealArray, mats: NDRealArray, vidxs: Optional[NDIntArray]=None,
           midxs: Optional[NDIntArray]=None, num_threads: int=1) -> NDRealArray:
    """Find a matrix product of two arrays ``vecs[i]`` and ``mats[j]``, where indices  are given by
    arguments ``vidxs`` and ``midxs``.

    Args:
        vecs : Array of vectors.
        mats : Array of matrices.
        vidxs : Vector indices.
        midxs : Matrix indices.
        num_threads : Number of threads used in the calculations.

    Returns:
        An array of matrix products.
    """
    ...

def matmul_vjp(cotangents: NDRealArray, vecs: NDRealArray, mats: NDRealArray,
               vidxs: Optional[NDIntArray]=None, midxs: Optional[NDIntArray]=None,
               num_threads: int=1) -> Tuple[NDRealArray, NDRealArray]:
    ...

def source_lines(q: NDRealArray, kmin: RealSequence, kmax: RealSequence,
                 num_threads: int=1) -> NDRealArray:
    r"""Calculate the source lines for a set of reciprocal lattice points ``q``.

    Args:
        q : Array of reciprocal lattice points.
        kin_min : Lower bound of the rectangular aperture function.
        kin_max : Upper bound of the rectangular aperture function.
        num_threads : A number of threads used in the computations.

    Returns:
        A set of source lines in the aperture function.
    """
    ...

def source_lines_vjp(cotangents: NDRealArray, kin: NDRealArray, q: NDRealArray,
                     kmin: RealSequence, kmax: RealSequence,
                     num_threads: int=1) -> NDRealArray:
    ...
