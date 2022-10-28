"""Convergent beam experimental setup consists of a crystalline sample, defined by position and
alignment (:class:`cbclib.Sample`) together with :class:`cbclib.Basis` unit cell basis vectors
that constitute an indexing solution. Each frame in a scan has it's own :class:`cbclib.Sample`,
and the collection of sample's position and alignment for each frame is stored in
:class:`cbclib.ScanSamples`.

Examples:
    Initialize a :class:`Sample` with a rotation matrix :class:`cbc.Rotation` and a coordinate
    array:

    >>> import cbclib as cbc
    >>> import numpy as np
    >>> sample = cbc.Sample(cbc.Rotation.import_tilt((0.1, 0.5 * np.pi, 0.5 * np.pi)), np.ones(3))
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from typing import (Any, ClassVar, Dict, ItemsView, Iterable, Iterator, KeysView, List, Optional,
                    Tuple, Union, ValuesView)
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel
from .bin import (tilt_matrix, cartesian_to_spherical, spherical_to_cartesian, euler_angles,
                  euler_matrix, tilt_angles, draw_line, draw_line_index)
from .cxi_protocol import Indices
from .data_container import DataContainer, INIContainer

@dataclass
class Basis(INIContainer):
    """An indexing solution, defined by a set of three unit cell vectors.

    Args:
        a_vec : First basis vector.
        b_vec : Second basis vector.
        c_vec : Third basis vector.
    """
    __ini_fields__ = {'basis': ('a_vec', 'b_vec', 'c_vec')}

    a_vec : np.ndarray
    b_vec : np.ndarray
    c_vec : np.ndarray

    def __post_init__(self):
        self.mat = np.stack((self.a_vec, self.b_vec, self.c_vec))

    @classmethod
    def import_matrix(cls, mat: np.ndarray) -> Basis:
        """Return a new :class:`Basis` object, initialised by a stacked matrix of three basis
        vectors.

        Args:
            mat : A matrix of three stacked basis vectors.

        Returns:
            A new :class:`Basis` object.
        """
        return cls(mat[0], mat[1], mat[2])

    @classmethod
    def import_spherical(cls, mat: np.ndarray) -> Basis:
        """Return a new :class:`Basis` object, initialised by a stacked matrix of three basis
        vectors written in spherical coordinate system.

        Args:
            mat : A matrix of three stacked basis vectors in spherical coordinate system.

        Returns:
            A new :class:`Basis` object.
        """
        return cls.import_matrix(spherical_to_cartesian(mat))

    def reciprocate(self, setup: ScanSetup) -> Basis:
        """Calculate a set of reciprocal unit cell vectors.

        Args:
            setup : Experimenat setup.

        Returns:
            A set of reciprocal unit cell vectors.
        """
        a_rec = np.cross(self.b_vec, self.c_vec) / (np.cross(self.b_vec, self.c_vec).dot(self.a_vec))
        b_rec = np.cross(self.c_vec, self.a_vec) / (np.cross(self.c_vec, self.a_vec).dot(self.b_vec))
        c_rec = np.cross(self.a_vec, self.b_vec) / (np.cross(self.a_vec, self.b_vec).dot(self.c_vec))
        return Basis.import_matrix(np.stack((a_rec, b_rec, c_rec)) * setup.wavelength)

    def to_spherical(self) -> np.ndarray:
        """Return a stack of unit cell vectors in spherical coordinate system.

        Returns:
            A matrix of three stacked unit cell vectors in spherical coordinate system.
        """
        return cartesian_to_spherical(self.mat)

@dataclass
class ScanSetup(INIContainer):
    """Convergent beam crystallography experimental setup. Contains all the important distances
    and characteristics of the experimental setup.

    Args:
        foc_pos : Focus position relative to the detector [m].
        rot_axis : Axis of rotation.
        pupil_min : Lower bound of the aperture function in the detector plane.
        pupil_max : Upper bound of the aperture function in the detector plane.
        wavelength : X-ray beam wavelength [m].
        x_pixel_size : Detector pixel size along the x axis [m].
        y_pixel_size : Detector pixel size along the y axis [m].
    """
    __ini_fields__ = {'exp_geom': ('foc_pos', 'rot_axis', 'pupil_min', 'pupil_max',
                                   'wavelength', 'x_pixel_size', 'y_pixel_size')}

    foc_pos         : np.ndarray
    rot_axis        : np.ndarray
    pupil_min       : np.ndarray
    pupil_max       : np.ndarray
    wavelength      : float
    x_pixel_size    : float
    y_pixel_size    : float

    def __post_init__(self):
        self.rot_axis = self.rot_axis / np.sqrt(np.sum(self.rot_axis**2))
        self.kin_min = self.detector_to_kin(self.pupil_min[0], self.pupil_min[1])
        self.kin_max = self.detector_to_kin(self.pupil_max[0], self.pupil_max[1])
        self.kin_center = self.detector_to_kin(0.5 * (self.pupil_min[0] + self.pupil_max[0]),
                                               0.5 * (self.pupil_min[1] + self.pupil_max[1]))

    def _det_to_k(self, x: np.ndarray, y: np.ndarray, source: np.ndarray) -> np.ndarray:
        delta_x = x * self.x_pixel_size - source[0]
        delta_y = y * self.y_pixel_size - source[1]
        phis = np.arctan2(delta_y, delta_x)
        thetas = np.arctan(np.sqrt(delta_x**2 + delta_y**2) / source[2])
        return np.stack((np.sin(thetas) * np.cos(phis),
                         np.sin(thetas) * np.sin(phis),
                         np.cos(thetas)), axis=-1)

    def _k_to_det(self, karr: np.ndarray, source: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        theta = np.arccos(karr[..., 2] / np.sqrt((karr * karr).sum(axis=-1)))
        phi = np.arctan2(karr[..., 1], karr[..., 0])
        det_x = source[2] * np.tan(theta) * np.cos(phi) + source[0]
        det_y = source[2] * np.tan(theta) * np.sin(phi) + source[1]
        return det_x / self.x_pixel_size, det_y / self.y_pixel_size

    def detector_to_kout(self, x: np.ndarray, y: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """Project detector coordinates ``(x, y)`` to the output wave-vectors space originating
        from the point ``pos``.

        Args:
            x : A set of x coordinates.
            y : A set of y coordinates.
            pos : Source point of the output wave-vectors.

        Returns:
            An array of output wave-vectors.
        """
        return self._det_to_k(x, y, pos)

    def kout_to_detector(self, kout: np.ndarray, pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project output wave-vectors originating from the point ``pos`` to the detector plane.

        Args:
            kout : Output wave-vectors.
            pos : Source point of the output wave-vectors.

        Returns:
            A tuple of x and y detector coordinates.
        """
        return self._k_to_det(kout, pos)

    def detector_to_kin(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Project detector coordinates ``(x, y)`` to the incident wave-vectors space.

        Args:
            x : A set of x coordinates.
            y : A set of y coordinates.

        Returns:
            An array of incident wave-vectors.
        """
        return self._det_to_k(x, y, self.foc_pos)

    def kin_to_detector(self, kin: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project incident wave-vectors to the detector plane.

        Args:
            kin : Incident wave-vectors.

        Returns:
            A tuple of x and y detector coordinates.
        """
        return self._k_to_det(kin, self.foc_pos)

    def tilt_rotation(self, theta: float) -> Rotation:
        """Return a tilt rotation by the angle ``theta`` arount the axis of rotation.

        Args:
            theta : Angle of rotation.

        Returns:
            A new :class:`cbclib.Rotation` object.
        """
        return Rotation.import_tilt((theta, np.arccos(self.rot_axis[2]),
                                     np.arctan2(self.rot_axis[1], self.rot_axis[0])))

FloatArray = Union[np.ndarray, List[float], Tuple[float, float, float]]

@dataclass
class Rotation(DataContainer):
    """A rotation matrix implementation. Provides auxiliary methods to work with Euler
    and tilt angles.

    Args:
        matrix : Rotation matrix.
    """
    matrix : np.ndarray = np.eye(3, 3)

    def __post_init__(self):
        self.matrix = self.matrix.reshape((3, 3))

    @classmethod
    def import_euler(cls, angles: FloatArray) -> Rotation:
        r"""Calculate a rotation matrix from Euler angles with Bunge convention [EUL]_.

        Args:
            angles : Euler angles :math:`\phi_1, \Phi, \phi_2`.

        Returns:
            A new rotation matrix :class:`Rotation`.
        """
        return Rotation(euler_matrix(np.asarray(angles)))

    @classmethod
    def import_tilt(cls, angles: FloatArray) -> Rotation:
        r"""Calculate a rotation matrix for a set of three angles set of three angles
        :math:`\theta, \alpha, \beta`, a rotation angle :math:`\theta`, an angle between the
        axis of rotation and OZ axis :math:`\alpha`, and a polar angle of the axis of rotation
        :math:`\beta`.

        Args:
            angles : A set of angles :math:`\theta, \alpha, \beta`.

        Returns:
            A new rotation matrix :class:`Rotation`.
        """
        return Rotation(tilt_matrix(np.asarray(angles)))

    def __call__(self, inp: np.ndarray) -> np.ndarray:
        """Apply the rotation to a set of vectors ``inp``.

        Args:
            inp : A set of 3D vectors.

        Returns:
            A set of rotated 3D vectors.
        """
        return inp.dot(self.matrix.T)

    def __mul__(self, obj: Any) -> Rotation:
        """Calculate a product of two rotations.

        Args:
            obj : A rotation matrix.

        Returns:
            A new rotation matrix that is a product of two rotations.
        """
        if isinstance(obj, Rotation):
            return Rotation(self.matrix.dot(obj.matrix))
        return NotImplemented

    def reciprocate(self) -> Rotation:
        """Invert the rotation matrix.

        Returns:
            An inverse rotation matrix.
        """
        return Rotation(self.matrix.T)

    def to_euler(self) -> np.ndarray:
        r"""Calculate Euler angles with Bunge convention [EUL]_.

        Returns:
            A set of Euler angles with Bunge convention :math:`\phi_1, \Phi, \phi_2`.
        """
        return euler_angles(self.matrix)

    def to_tilt(self) -> np.ndarray:
        r"""Calculate an axis of rotation and a rotation angle for a rotation matrix.

        Returns:
            A set of three angles :math:`\theta, \alpha, \beta`, a rotation angle :math:`\theta`,
            an angle between the axis of rotation and OZ axis :math:`\alpha`, and a polar angle
            of the axis of rotation :math:`\beta`.
        """
        if np.allclose(self.matrix, self.matrix.T):
            eigw, eigv = np.linalg.eigh(self.matrix)
            axis = eigv[np.isclose(eigw, 1.0)]
            theta = np.arccos(0.5 * (np.trace(self.matrix) - 1.0))
            return np.array([theta, np.arccos(axis[0, 2]), np.arctan2(axis[0, 1], axis[0, 0])])
        return tilt_angles(self.matrix)

@dataclass
class Sample():
    """A convergent beam sample implementation. Stores position and orientation of the sample.

    Args:
        rotation : rotation matrix, that defines the orientation of the sample.
        position : Sample's position [m].
    """
    rotation : Rotation
    position : np.ndarray
    mat_columns : ClassVar[Tuple[str]] = ('Rxx', 'Rxy', 'Rxz',
                                          'Ryx', 'Ryy', 'Ryz',
                                          'Rzx', 'Rzy', 'Rzz')
    pos_columns : ClassVar[Tuple[str]] = ('x', 'y', 'z')

    @classmethod
    def import_dataframe(cls, data: pd.Series) -> Sample:
        """Initialize a new :class:`Sample` object with a :class:`pandas.Series` array. The array
        must contain the following columns:
        * 'Rxx', 'Rxy', 'Rxz', 'Ryx', 'Ryy', 'Ryz', 'Rzx', 'Rzy', 'Rzz' : Rotational matrix.
        * 'x', 'y', 'z' : Sample's position [m].

        Args:
            data : A :class:`pandas.Series` array.

        Returns:
            A new :class:`Sample` object.
        """
        return cls(rotation=Rotation(data[list(cls.mat_columns)].to_numpy()),
                   position=data[list(cls.pos_columns)].to_numpy())

    def rotate(self, basis: Basis) -> Basis:
        """Rotate a :class:`cbclib.Basis` by the ``rotation`` attribute.

        Args:
            basis : Indexing solution basis vectors.

        Returns:
            A new rotated :class:`cbclib.Basis` object.
        """
        return Basis.import_matrix(self.rotation(basis.mat))

    def detector_to_kout(self, x: np.ndarray, y: np.ndarray, setup: ScanSetup) -> np.ndarray:
        """Project detector coordinates ``(x, y)`` to the output wave-vectors space originating
        from the sample's position.

        Args:
            x : A set of x coordinates.
            y : A set of y coordinates.
            setup : Experimental setup.

        Returns:
            An array of output wave-vectors.
        """
        return setup.detector_to_kout(x, y, self.position)

    def kout_to_detector(self, kout: np.ndarray, setup: ScanSetup) -> Tuple[np.ndarray, np.ndarray]:
        """Project output wave-vectors originating from the sample's position to the detector
        plane.

        Args:
            kout : Output wave-vectors.
            setup : Experimental setup.

        Returns:
            A tuple of x and y detector coordinates.
        """
        return setup.kout_to_detector(kout, self.position)

    def to_dataframe(self) -> pd.Series:
        """Export the sample object to a :class:`pandas.Series` array.

        Returns:
            A :class:`pandas.Series` array with the following columns:
            * 'Rxx', 'Rxy', 'Rxz', 'Ryx', 'Ryy', 'Ryz', 'Rzx', 'Rzy', 'Rzz' : Rotational
              matrix.
            * 'x', 'y', 'z' : Sample's position [m].
        """
        return pd.Series(np.concatenate((self.rotation.matrix.ravel(), self.position)),
                         index=self.mat_columns + self.pos_columns)

MapSamples = Union[Iterable[Tuple[int, Sample]], Dict[int, Sample]]

class ScanSamples():
    """A collection of scan :class:`cbclib.Sample` objects. Provides an interface to import
    from and exprort to a :class:`pandas.DataFrame` table and a set of dictionary methods.
    """
    def __init__(self, items: MapSamples=list()):
        self._dct = dict(items)

    @classmethod
    def import_dataframe(cls, df: pd.DataFrame) -> ScanSamples:
        """Initialize a new :class:`ScanSamples` container with a :class:`pandas.DataFrame`
        table. The table must contain the following columns:
        * 'Rxx', 'Rxy', 'Rxz', 'Ryx', 'Ryy', 'Ryz', 'Rzx', 'Rzy', 'Rzz' : Rotational matrix.
        * 'x', 'y', 'z' : Sample's position [m].

        Args:
            data : A :class:`pandas.DataFrame` table.

        Returns:
            A new :class:`ScanSamples` container.
        """
        frames = df.index
        samples = df.apply(Sample.import_dataframe, axis=1)
        return cls(zip(frames, samples))

    def __getitem__(self, frame: int) -> Sample:
        return self._dct.__getitem__(frame)

    def __setitem__(self, frame: int, smp: Sample):
        self._dct.__setitem__(frame, smp)

    def __iter__(self) -> Iterator[int]:
        return self._dct.__iter__()

    def __contains__(self, frame: int) -> bool:
        return self._dct.__contains__(frame)

    def __copy__(self) -> ScanSamples:
        return ScanSamples(self)

    def __deepcopy__(self, memo: Dict[int, Any]) -> ScanSamples:
        return ScanSamples({key: deepcopy(val, memo) for key, val in self.items()})

    def __str__(self) -> str:
        return self._dct.__str__()

    def __repr__(self) -> str:
        return self._dct.__repr__()

    def __len__(self) -> int:
        return len(self._dct)

    def keys(self) -> KeysView[str]:
        """Return a list of sample indices available in the container.

        Returns:
            List of sample indices available in the container.
        """
        return self._dct.keys()

    def values(self) -> ValuesView[Sample]:
        """Return a set of samples stored in the container.

        Returns:
            List of samples stored in the container.
        """
        return self._dct.values()

    def items(self) -> ItemsView[str, Sample]:
        """Return ``(index, sample)`` pairs stored in the container.

        Returns:
            ``(index, sample)`` pairs stored in the container.
        """
        return self._dct.items()

    def get_positions(self, axis: int) -> np.ndarray:
        """Return an array of sample coordinates along the given axis.

        Args:
            axis : Axis index in (0 - 2) range.

        Returns:
            Set of sample coordinates.
        """
        return np.asarray([sample.position[axis] for sample in self.values()])

    def regularise(self, kernel_bandwidth: Tuple[int, int]) -> ScanSamples:
        """Regularise sample positions by applying a Gaussian Process with a gaussian kernel
        bandwidth in the given interval.

        Args:
            kernel_bandwidth : Inverval of gaussian kernel bandwidths to use.

        Returns:
            A new :class:`ScanSamples` container with regularised sample positions.
        """
        obj = deepcopy(self)
        kernel = ConstantKernel(0.1, (1e-3, 0.5)) * RBF(kernel_bandwidth[0], kernel_bandwidth) + \
                 WhiteKernel(0.9, (0.5, 1.0))
        model = GaussianProcessRegressor(kernel, n_restarts_optimizer=10, normalize_y=True)
        frames = np.array(list(self.keys()))[:, None]
        for axis in range(3):
            model.fit(frames, obj.get_positions(axis=axis))
            obj = obj.set_positions(model.predict(frames), axis=axis)
        return obj

    def rotation(self, from_frames: Union[int, Iterable[int]],
                 to_frames: Union[int, Iterable[int]]) -> List[Rotation]:
        """Find a rotation that rotates ``from_frames`` samples to ``to_frames``.

        Args:
            from_frames : A set of indices of the samples from which rotations are calculated.
            to_frames : A set of indices of the sample to which rotations are calculated.

        Returns:
            A set of rotations.
        """
        from_frames, to_frames = np.atleast_1d(from_frames), np.atleast_1d(to_frames)
        rotations = []
        for f1, f2 in zip(from_frames, to_frames):
            rotations.append(self[f2].rotation * self[f1].rotation.reciprocate())
        return rotations

    def set_positions(self, positions: np.ndarray, axis: int) -> ScanSamples:
        """Update sample coordinates along the given axis.

        Args:
            axis : Axis index in (0 - 2) range.

        Returns:
            A new :class:`ScanSamples` container with updated sample positions.
        """
        obj = deepcopy(self)
        for frame in obj:
            obj[frame].position[axis] = positions[frame]
        return obj

    def to_dict(self) -> Dict[str, Sample]:
        """Export the sample container to a :class:`dict`.

        Returns:
            A dictionary of :class:`Sample` objects.
        """
        return dict(self._dct)

    def to_dataframe(self) -> pd.DataFrame:
        """Export the sample object to a :class:`pandas.DataFrame` table.

        Returns:
            A :class:`pandas.DataFrame` table with the following columns:
            * 'Rxx', 'Rxy', 'Rxz', 'Ryx', 'Ryy', 'Ryz', 'Rzx', 'Rzy', 'Rzz' : Rotational
              matrices.
            * 'x', 'y', 'z' : Sample positions [m].
        """
        return pd.DataFrame((sample.to_dataframe() for sample in self.values()), index=self.keys())

@dataclass
class Streaks(DataContainer):
    """Detector streak lines container. Provides an interface to draw a pattern for a set of
    lines.

    Args:
        x0 : x coordinates of the first point of a line.
        y0 : y coordinates of the first point of a line.
        x1 : x coordinates of the second point of a line.
        y1 : y coordinates of the second point of a line.
        width : Line's width in pixels.
        length: Line's length in pixels.
        h : First Miller index.
        k : Second Miller index.
        l : Third Miller index.
    """
    x0          : np.ndarray
    y0          : np.ndarray
    x1          : np.ndarray
    y1          : np.ndarray
    width       : np.ndarray
    length      : Optional[np.ndarray] = None
    h           : Optional[np.ndarray] = None
    k           : Optional[np.ndarray] = None
    l           : Optional[np.ndarray] = None

    @property
    def hkl(self) -> Optional[np.ndarray]:
        if self.h is None:
            return None
        return np.stack((self.h, self.k, self.l), axis=1)

    def __post_init__(self):
        if self.length is None:
            self.length = np.sqrt((self.x1 - self.x0)**2 + (self.y1 - self.y0)**2)

    def __len__(self) -> int:
        return self.length.shape[0]

    def mask_streaks(self, indices: Indices) -> Streaks:
        """Return a new streaks container with a set of streaks discarded.

        Args:
            indices : A set of indices of the streaks to discard.

        Returns:
            A new :class:`cbclib.Streaks` container.
        """
        return Streaks(**{attr: self[attr][indices] for attr in self.contents()})

    def pattern_dataframe(self, shape: Optional[Tuple[int, int]]=None, dp: float=1e-3, dilation: float=0.0,
                          profile: str='tophat', drop_duplicates: bool=False) -> pd.DataFrame:
        """Draw a pattern in the :class:`pandas.DataFrame` format.

        Args:
            shape : Detector grid shape.
            dp : Likelihood value increment.
            dilation : Line mask dilation in pixels.
            profile : Line width profiles. The following keyword values are allowed:

                * `tophat` : Top-hat (rectangular) function profile.
                * `linear` : Linear (triangular) function profile.
                * `quad` : Quadratic (parabola) function profile.

            drop_duplicates : Discard pixel data with duplicate x and y coordinates if True.

        Returns:
            A pattern in :class:`pandas.DataFrame` format.
        """
        df = pd.DataFrame(self.pattern_dict(shape, dp=dp, dilation=dilation, profile=profile))
        df = df[df['p'] > 0.0]
        if drop_duplicates:
            return df.drop_duplicates(['x', 'y'])
        return df

    def pattern_dict(self, shape: Optional[Tuple[int, int]]=None, dp: float=1e-3,
                     dilation: float=0.0, profile: str='tophat') -> Dict[str, np.ndarray]:
        """Draw a pattern in the :class:`dict` format.

        Args:
            shape : Detector grid shape.
            dp : Likelihood value increment.
            dilation : Line mask dilation in pixels.
            profile : Line width profiles. The following keyword values are allowed:

                * `tophat` : Top-hat (rectangular) function profile.
                * `linear` : Linear (triangular) function profile.
                * `quad` : Quadratic (parabola) function profile.

        Returns:
            A pattern in dictionary format.
        """
        if dp > 1.0 or dp <= 0.0:
            raise ValueError('`dp` must be in the range of (0.0, 1.0]')
        idx, x, y, p = draw_line_index(lines=self.to_numpy(), shape=shape, max_val=int(1.0 / dp),
                                       dilation=dilation, profile=profile).T
        pattern = {'index': idx, 'x': x, 'y': y, 'p': p / int(1.0 / dp)}
        for attr in ['h', 'k', 'l']:
            if attr in self.contents():
                pattern[attr] = self[attr][idx]
        return pattern

    def pattern_image(self, shape: Tuple[int, int], dp: float=1e-3, dilation: float=0.0,
                      profile: str='tophat') -> np.ndarray:
        """Draw a pattern in the :class:`numpy.ndarray` format.

        Args:
            shape : Detector grid shape.
            dp : Likelihood value increment.
            dilation : Line mask dilation in pixels.
            profile : Line width profiles. The following keyword values are allowed:

                * `tophat` : Top-hat (rectangular) function profile.
                * `linear` : Linear (triangular) function profile.
                * `quad` : Quadratic (parabola) function profile.

        Returns:
            A pattern in :class:`numpy.ndarray` format.
        """
        if dp > 1.0 or dp <= 0.0:
            raise ValueError('`dp` must be in the range of (0.0, 1.0]')
        mask = self.pattern_mask(shape, int(1.0 / dp), dilation, profile)
        return mask / int(1.0 / dp)

    def pattern_mask(self, shape: Tuple[int, int], max_val: int=1, dilation: float=0.0,
                     profile: str='tophat') -> np.ndarray:
        """Draw a pattern mask.

        Args:
            shape : Detector grid shape.
            max_val : Mask maximal value.
            dilation : Line mask dilation in pixels.
            profile : Line width profiles. The following keyword values are allowed:

                * `tophat` : Top-hat (rectangular) function profile.
                * `linear` : Linear (triangular) function profile.
                * `quad` : Quadratic (parabola) function profile.

        Returns:
            A pattern mask.
        """
        mask = np.zeros(shape, dtype=np.uint32)
        return draw_line(mask, lines=self.to_numpy(), max_val=max_val, dilation=dilation,
                         profile=profile)

    def to_dataframe(self) -> pd.DataFrame:
        """Export a streaks container into :class:`pandas.DataFrame`.

        Returns:
            A dataframe with all the data specified in :class:`cbclib.Streaks`.
        """
        return pd.DataFrame(dict(self))

    def to_numpy(self) -> np.ndarray:
        """Export a streaks container into :class:`numpy.ndarray`.

        Returns:
            An array with all the data specified in :class:`cbclib.Streaks`.
        """
        return np.stack([self[attr] for attr in self.contents()], axis=1)
