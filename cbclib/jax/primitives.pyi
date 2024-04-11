from typing import Optional
from ..annotations import JaxIntArray, JaxRealArray, RealSequence

def euler_angles(rmats: JaxRealArray, num_threads: int=1) -> JaxRealArray:
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

def euler_matrix(angles: JaxRealArray, num_threads: int=1) -> JaxRealArray:
    r"""Calculate rotation matrices from Euler angles with Bunge convention [EUL]_.

    Args:
        angles : Euler angles :math:`\phi_1, \Phi, \phi_2`.
        num_threads : Number of threads used in the calculations.

    Returns:
        A set of rotation matrices.
    """
    ...


def tilt_angles(rmats: JaxRealArray, num_threads: int=1) -> JaxRealArray:
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

def tilt_matrix(angles: JaxRealArray) -> JaxRealArray:
    r"""Calculate a rotation matrix for a set of three angles set of three angles :math:`\theta,
    \alpha, \beta`, a rotation angle :math:`\theta`, an angle between the axis of rotation and
    OZ axis :math:`\alpha`, and a polar angle of the axis of rotation :math:`\beta`.

    Args:
        angles : A set of angles :math:`\theta, \alpha, \beta`.

    Returns:
        A set of rotation matrices.
    """
    ...

def det_to_k(x: JaxRealArray, y: JaxRealArray, src: JaxRealArray,
             idxs: Optional[JaxIntArray]=None, num_threads: int=1) -> JaxRealArray:
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

def k_to_det(k: JaxRealArray, src: JaxRealArray, idxs: Optional[JaxIntArray]=None,
             num_threads: int=1) -> JaxRealArray:
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

def k_to_smp(k: JaxRealArray, z: JaxRealArray, src: RealSequence,
             idxs: Optional[JaxIntArray]=None, num_threads: int=1) -> JaxRealArray:
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

def matmul(vecs: JaxRealArray, mats: JaxRealArray, vidxs: Optional[JaxIntArray]=None,
           midxs: Optional[JaxIntArray]=None, num_threads: int=1) -> JaxRealArray:
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

def source_lines(q: JaxRealArray, kmin: RealSequence, kmax: RealSequence,
                 num_threads: int=1) -> JaxRealArray:
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

def draw_line(out: JaxRealArray, lines: JaxRealArray, idxs: Optional[JaxIntArray]=None,
              max_val: float=1.0, kernel: str='biweight', num_threads: int=1) -> JaxRealArray:
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
