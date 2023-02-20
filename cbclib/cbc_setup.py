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
from dataclasses import dataclass, fields
from typing import (Any, ClassVar, Dict, ItemsView, Sequence, Iterator, KeysView, List, Optional,
                    Tuple, Union, ValuesView)
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel
from .bin import (tilt_matrix, cartesian_to_spherical, spherical_to_cartesian, euler_angles,
                  det_to_k, k_to_det, euler_matrix, tilt_angles, draw_line_image, draw_line_mask,
                  draw_line_table, calc_source_lines, filter_hkl)
from .cxi_protocol import Indices
from .data_container import DataContainer, INIContainer, Transform, Crop

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

    def generate_hkl(self, q_abs: float) -> np.ndarray:
        """Return a set of reflections lying inside of a sphere of radius ``q_abs`` in the
        reciprocal space.

        Args:
            q_abs : The radius of a sphere in the reciprocal space.

        Returns:
            An array of Miller indices, that lie inside of the sphere.
        """
        lat_size = np.rint(q_abs / self.to_spherical()[:, 0]).astype(int)
        h_idxs = np.arange(-lat_size[0], lat_size[0] + 1)
        k_idxs = np.arange(-lat_size[1], lat_size[1] + 1)
        l_idxs = np.arange(-lat_size[2], lat_size[2] + 1)
        h_grid, k_grid, l_grid = np.meshgrid(h_idxs, k_idxs, l_idxs)
        hkl = np.stack((h_grid.ravel(), k_grid.ravel(), l_grid.ravel()), axis=1)
        hkl = np.compress(hkl.any(axis=1), hkl, axis=0)

        rec_vec = hkl.dot(self.mat)
        rec_abs = np.sqrt((rec_vec**2).sum(axis=-1))
        return hkl[rec_abs < q_abs]

    def lattice_constants(self) -> np.ndarray:
        r"""Return lattice constants :math:`a, b, c, \alpha, \beta, \gamma`. The unit cell
        length are unitless.

        Returns:
            An array of lattice constants.
        """
        lengths = self.to_spherical()[:, 0]
        alpha = np.arccos(np.sum(self.mat[1] * self.mat[2]) / (lengths[1] * lengths[2]))
        beta = np.arccos(np.sum(self.mat[0] * self.mat[2]) / (lengths[0] * lengths[2]))
        gamma = np.arccos(np.sum(self.mat[0] * self.mat[1]) / (lengths[0] * lengths[1]))
        return np.concatenate((lengths, [alpha, beta, gamma]))

    def reciprocate(self) -> Basis:
        """Calculate the basis of the reciprocal lattice.

        Returns:
            The basis of the reciprocal lattice.
        """
        a_rec = np.cross(self.b_vec, self.c_vec) / (np.cross(self.b_vec, self.c_vec).dot(self.a_vec))
        b_rec = np.cross(self.c_vec, self.a_vec) / (np.cross(self.c_vec, self.a_vec).dot(self.b_vec))
        c_rec = np.cross(self.a_vec, self.b_vec) / (np.cross(self.a_vec, self.b_vec).dot(self.c_vec))
        return Basis.import_matrix(np.stack((a_rec, b_rec, c_rec)))

    def to_spherical(self) -> np.ndarray:
        """Return a stack of unit cell vectors in spherical coordinate system.

        Returns:
            A matrix of three stacked unit cell vectors in spherical coordinate system.
        """
        return cartesian_to_spherical(self.mat)

@dataclass
class ScanSetup(INIContainer):
    """Convergent beam crystallography experimental setup. Contains the parameters of the scattering
    geometry and experimental setup.

    Args:
        foc_pos : Focus position relative to the detector [m].
        pupil_roi : Region of interest of the aperture function in the detector plane. Comprised
            of four elements ``[y_min, y_max, x_min, x_max]``.
        rot_axis : Axis of rotation.
        smp_dist : Focus-to-sample distance [m].
        wavelength : X-ray beam wavelength [m].
        x_pixel_size : Detector pixel size along the x axis [m].
        y_pixel_size : Detector pixel size along the y axis [m].
    """
    __ini_fields__ = {'exp_geom': ('foc_pos', 'pupil_roi', 'rot_axis', 'smp_dist', 'wavelength',
                                   'x_pixel_size', 'y_pixel_size')}

    foc_pos         : np.ndarray
    pupil_roi       : np.ndarray
    rot_axis        : np.ndarray
    smp_dist        : float
    wavelength      : float
    x_pixel_size    : float
    y_pixel_size    : float

    def __post_init__(self):
        self.kin_min = self.detector_to_kin(x=self.pupil_roi[2], y=self.pupil_roi[0])[0]
        self.kin_max = self.detector_to_kin(x=self.pupil_roi[3], y=self.pupil_roi[1])[0]
        self.kin_center = self.detector_to_kin(x=np.mean(self.pupil_roi[2:]),
                                               y=np.mean(self.pupil_roi[:2]))[0]

    def detector_to_kout(self, x: np.ndarray, y: np.ndarray, pos: np.ndarray,
                         num_threads: int=1) -> np.ndarray:
        """Project detector coordinates ``(x, y)`` to the output wave-vectors space originating
        from the point ``pos``.

        Args:
            x : A set of x coordinates.
            y : A set of y coordinates.
            pos : Source point of the output wave-vectors.
            num_threads : Number of threads used in the calculations.

        Returns:
            An array of output wave-vectors.
        """
        return det_to_k(x=np.atleast_1d(x), y=np.atleast_1d(y), src=pos,
                        x_ps=self.x_pixel_size, y_ps=self.y_pixel_size, num_threads=num_threads)

    def kout_to_detector(self, kout: np.ndarray, pos: np.ndarray,
                         num_threads: int=1) -> Tuple[np.ndarray, np.ndarray]:
        """Project output wave-vectors originating from the point ``pos`` to the detector plane.

        Args:
            kout : Output wave-vectors.
            pos : Source point of the output wave-vectors.
            num_threads : Number of threads used in the calculations.

        Returns:
            A tuple of x and y detector coordinates.
        """
        det_x, det_y = k_to_det(karr=kout, src=pos, num_threads=num_threads)
        return det_x / self.x_pixel_size, det_y / self.y_pixel_size

    def detector_to_kin(self, x: np.ndarray, y: np.ndarray, num_threads: int=1) -> np.ndarray:
        """Project detector coordinates ``(x, y)`` to the incident wave-vectors space.

        Args:
            x : A set of x coordinates.
            y : A set of y coordinates.
            num_threads : Number of threads used in the calculations.

        Returns:
            An array of incident wave-vectors.
        """
        return det_to_k(x=np.atleast_1d(x), y=np.atleast_1d(y), src=self.foc_pos,
                        x_ps=self.x_pixel_size, y_ps=self.y_pixel_size, num_threads=num_threads)

    def kin_to_detector(self, kin: np.ndarray, num_threads: int=1) -> Tuple[np.ndarray, np.ndarray]:
        """Project incident wave-vectors to the detector plane.

        Args:
            kin : Incident wave-vectors.
            num_threads : Number of threads used in the calculations.

        Returns:
            A tuple of x and y detector coordinates.
        """
        det_x, det_y = k_to_det(karr=kin, src=self.foc_pos, num_threads=num_threads)
        return det_x / self.x_pixel_size, det_y / self.y_pixel_size

    def kin_to_sample(self, kin: np.ndarray, z: Optional[float]=None,
                      num_threads: int=1) -> Tuple[np.ndarray, np.ndarray]:
        """Project incident wave-vectors to the detector plane.

        Args:
            kin : Incident wave-vectors.
            z : z coordinate of the sample [m].
            num_threads : Number of threads used in the calculations.

        Returns:
            A tuple of x and y detector coordinates.
        """
        if z is None:
            z = self.foc_pos[2] + self.smp_dist
        source = np.array([self.foc_pos[0], self.foc_pos[1], self.foc_pos[2] - z])
        return k_to_det(karr=kin, src=source, num_threads=num_threads)

    def tilt_rotation(self, theta: float) -> Rotation:
        """Return a tilt rotation by the angle ``theta`` arount the axis of rotation.

        Args:
            theta : Angle of rotation.

        Returns:
            A new :class:`cbclib.Rotation` object.
        """
        return Rotation.import_tilt((theta, self.rot_axis[0], self.rot_axis[1]))

    def tilt_samples(self, frames: np.ndarray, thetas: np.ndarray) -> ScanSamples:
        """Return a list of sample position and orientations of a tilt series.

        Args:
            frames : Set of frame indices.
            thetas : Set of sample tilts.

        Returns:
            A container of sample objects :class:`ScanSamples`.
        """
        angles = np.empty((thetas.size, 3))
        angles[:, 0] = thetas
        angles[:, 1:] = self.rot_axis
        rmats = tilt_matrix(angles).reshape(-1, 3, 3)
        return ScanSamples({frame: Sample(Rotation(rmat), self.foc_pos[2] + self.smp_dist)
                            for frame, rmat in zip(frames, rmats)})

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
    z : float
    mat_columns : ClassVar[Tuple[str]] = ('Rxx', 'Rxy', 'Rxz',
                                          'Ryx', 'Ryy', 'Ryz',
                                          'Rzx', 'Rzy', 'Rzz')
    z_column : ClassVar[str] = 'z'

    def __post_init__(self):
        if isinstance(self.z, np.ndarray):
            self.z = self.z.item()

    @classmethod
    def import_dataframe(cls, data: pd.Series) -> Sample:
        """Initialize a new :class:`Sample` object with a :class:`pandas.Series` array. The array
        must contain the following columns:

        * `Rxx`, `Rxy`, `Rxz`, `Ryx`, `Ryy`, `Ryz`, `Rzx`, `Rzy`, `Rzz` : Rotational matrix.
        * `z` : z coordinate [m].

        Args:
            data : A :class:`pandas.Series` array.

        Returns:
            A new :class:`Sample` object.
        """
        return cls(rotation=Rotation(data[list(cls.mat_columns)].to_numpy()),
                   z=data[cls.z_column])

    def position(self, setup: ScanSetup) -> np.ndarray:
        """Return sample coordinates relative to the detector.

        Args:
            setup : Experimental setup.

        Returns:
            An array of x, y, and z sample coordinates.
        """
        smp_x, smp_y = setup.kin_to_sample(setup.kin_center, self.z)
        return np.array([smp_x, smp_y, self.z])

    def rotate(self, basis: Basis) -> Basis:
        """Rotate a :class:`cbclib.Basis` by the ``rotation`` attribute.

        Args:
            basis : Indexing solution basis vectors.

        Returns:
            A new rotated :class:`cbclib.Basis` object.
        """
        return Basis.import_matrix(self.rotation(basis.mat))

    def replace(self, **kwargs: Any) -> Sample:
        """Return a new :class:`Sample` object with a set of attributes replaced.

        Args:
            kwargs : A set of attributes and the values to to replace.

        Returns:
            A new :class:`Sample` object with the updated attributes.
        """
        return Sample(**dict(self.to_dict(), **kwargs))

    def to_dict(self) -> Dict[str, Any]:
        """Export the :class:`Sample` object to a :class:`dict`.

        Returns:
            A dictionary of :class:`Sample` object's attributes.
        """
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def detector_to_kout(self, x: np.ndarray, y: np.ndarray, setup: ScanSetup,
                         rec_vec: Optional[np.ndarray]=None, num_threads: int=1) -> np.ndarray:
        """Project detector coordinates ``(x, y)`` to the output wave-vectors space originating
        from the sample's position.

        Args:
            x : A set of x coordinates.
            y : A set of y coordinates.
            setup : Experimental setup.
            rec_vec : A set of scattering vectors corresponding to the detector points.
            num_threads : Number of threads used in the calculations.

        Returns:
            An array of output wave-vectors.
        """
        if rec_vec is None:
            return setup.detector_to_kout(x, y, self.position(setup), num_threads)
        kout = setup.detector_to_kout(x, y, self.position(setup), num_threads)
        smp_x, smp_y = setup.kin_to_sample(kout - rec_vec, self.z, num_threads)
        smp_pos = np.stack((smp_x, smp_y, self.z * np.ones(smp_x.shape)), axis=-1)
        return setup.detector_to_kout(x, y, smp_pos, num_threads)

    def kout_to_detector(self, kout: np.ndarray, setup: ScanSetup, rec_vec: Optional[np.ndarray]=None,
                         num_threads: int=1) -> Tuple[np.ndarray, np.ndarray]:
        """Project output wave-vectors originating from the sample's position to the detector
        plane.

        Args:
            kout : Output wave-vectors.
            setup : Experimental setup.
            rec_vec : A set of scattering vectors corresponding to the output wave-vectors.
            num_threads : Number of threads used in the calculations.

        Returns:
            A tuple of x and y detector coordinates.
        """
        if rec_vec is None:
            return setup.kout_to_detector(kout, self.position(setup), num_threads)
        smp_x, smp_y = setup.kin_to_sample(kout - rec_vec, self.z, num_threads)
        smp_pos = np.stack((smp_x, smp_y, self.z * np.ones(smp_x.shape)), axis=-1)
        return setup.kout_to_detector(kout, smp_pos, num_threads)

    def to_dataframe(self) -> pd.Series:
        """Export the sample object to a :class:`pandas.Series` array.

        Returns:
            A :class:`pandas.Series` array with the following columns:

            * `Rxx`, `Rxy`, `Rxz`, `Ryx`, `Ryy`, `Ryz`, `Rzx`, `Rzy`, `Rzz` : Rotational
              matrix.
            * `z` : z coordinate [m].
        """
        return pd.Series(np.append(self.rotation.matrix.ravel(), self.z),
                         index=self.mat_columns + (self.z_column,))

MapSamples = Union[Sequence[Tuple[int, Sample]], Dict[int, Sample]]

class ScanSamples():
    """A collection of sample :class:`cbclib.Sample` objects. Provides an interface to import
    from and exprort to a :class:`pandas.DataFrame` table and a set of dictionary methods.
    """
    def __init__(self, items: MapSamples=list()):
        self._dct = dict(items)

    @classmethod
    def import_dataframe(cls, df: pd.DataFrame) -> ScanSamples:
        """Initialize a new :class:`ScanSamples` container with a :class:`pandas.DataFrame`
        table. The table must contain the following columns:

        * `Rxx`, `Rxy`, `Rxz`, `Ryx`, `Ryy`, `Ryz`, `Rzx`, `Rzy`, `Rzz` : Rotational matrix.
        * `z` : z coordinate [m].

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

    def positions(self, setup: ScanSetup) -> np.ndarray:
        """Return an array of sample coordinates relative to the detector.

        Args:
            setup : Experimental setup.

        Returns:
            Array of sample coordinates.
        """
        return np.stack([sample.position(setup) for sample in self.values()])

    def rotations(self) -> np.ndarray:
        """Return an array of sample rotations.

        Returns:
            Array of rotation matrices.
        """
        return np.stack([sample.rotation.matrix for sample in self.values()])

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
        model.fit(frames, np.array([sample.z for sample in self.values()]))
        for frame, z in zip(obj, model.predict(frames)):
            obj[frame] = obj[frame].replace(z=z)
        return obj

    def find_rotations(self, from_frames: Union[int, Sequence[int]],
                       to_frames: Union[int, Sequence[int]]) -> Union[Rotation, List[Rotation]]:
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
        if len(rotations) == 1:
            return rotations[0]
        return rotations

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

            * `Rxx`, `Rxy`, `Rxz`, `Ryx`, `Ryy`, `Ryz`, `Rzx`, `Rzy`, `Rzz` : Rotational
              matrix.
            * `z` : z coordinate [m].
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

    def pattern_dataframe(self, shape: Optional[Tuple[int, int]]=None, dilation: float=0.0,
                          profile: str='tophat', reduce: bool=True) -> pd.DataFrame:
        """Draw a pattern in the :class:`pandas.DataFrame` format.

        Args:
            shape : Detector grid shape.
            dilation : Line mask dilation in pixels.
            profile : Line width profiles. The following keyword values are allowed:

                * `tophat` : Top-hat (rectangular) function profile.
                * `linear` : Linear (triangular) function profile.
                * `quad` : Quadratic (parabola) function profile.
                * `gauss` : Gaussian function profile.

            reduce : Discard the pixel data with reflection profile values equal to
                zero.

        Returns:
            A pattern in :class:`pandas.DataFrame` format.
        """
        df = pd.DataFrame(self.pattern_dict(shape, dilation=dilation, profile=profile))
        if reduce:
            return df[df['rp'] > 0.0]
        return df

    def pattern_dict(self, shape: Optional[Tuple[int, int]]=None, dilation: float=0.0,
                     profile: str='tophat') -> Dict[str, np.ndarray]:
        """Draw a pattern in the :class:`dict` format.

        Args:
            shape : Detector grid shape.
            dilation : Line mask dilation in pixels.
            profile : Line width profiles. The following keyword values are allowed:

                * `tophat` : Top-hat (rectangular) function profile.
                * `linear` : Linear (triangular) function profile.
                * `quad` : Quadratic (parabola) function profile.
                * `gauss` : Gaussian function profile.

        Returns:
            A pattern in dictionary format.
        """
        idx, x, y, rp = draw_line_table(lines=self.to_numpy(), shape=shape, dilation=dilation,
                                        profile=profile)
        pattern = {'index': idx, 'x': x, 'y': y, 'rp': rp}
        for attr in ['h', 'k', 'l']:
            if attr in self.contents():
                pattern[attr] = self[attr][idx]
        return pattern

    def pattern_image(self, shape: Tuple[int, int], dilation: float=0.0,
                      profile: str='gauss') -> np.ndarray:
        """Draw a pattern in the :class:`numpy.ndarray` format.

        Args:
            shape : Detector grid shape.
            dilation : Line mask dilation in pixels.
            profile : Line width profiles. The following keyword values are allowed:

                * `tophat` : Top-hat (rectangular) function profile.
                * `linear` : Linear (triangular) function profile.
                * `quad` : Quadratic (parabola) function profile.
                * `gauss` : Gaussian function profile.

        Returns:
            A pattern in :class:`numpy.ndarray` format.
        """
        return draw_line_image(shape, lines=self.to_numpy(), dilation=dilation,
                               profile=profile)

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
                * `gauss` : Gaussian function profile.

        Returns:
            A pattern mask.
        """
        return draw_line_mask(shape, lines=self.to_numpy(), max_val=max_val,
                              dilation=dilation, profile=profile)

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

@dataclass
class CBDModel():
    """Prediction model for Convergent Beam Diffraction (CBD) pattern. The method uses the
    geometrical schematic of CBD diffraction in the reciprocal space [CBM]_ to predict a CBD
    pattern for the given crystalline sample.

    Args:
        basis : Unit cell basis vectors.
        sample : Sample position and orientation.
        setup : Experimental setup.
        transform : Any of the image transform objects.
        shape : Shape of the detector pixel grid.

    References:
        .. [CBM] Ho, Joseph X et al. “Convergent-beam method in macromolecular crystallography”,
                 Acta crystallographica Section D, Biological crystallography vol. 58, Pt. 12
                 (2002): 2087-95, https://doi.org/10.1107/s0907444902017511.
    """
    basis       : Basis
    sample      : Sample
    setup       : ScanSetup
    transform   : Optional[Transform] = None
    shape       : Optional[Tuple[int, int]] = None

    def __post_init__(self):
        self.basis = self.sample.rotate(self.basis)
        if isinstance(self.transform, Crop):
            self.shape = (self.transform.roi[1] - self.transform.roi[0],
                          self.transform.roi[3] - self.transform.roi[2])

    def filter_hkl(self, hkl: np.ndarray) -> np.ndarray:
        """Return a set of reciprocal lattice points that lie in the region of reciprocal space
        involved in diffraction.

        Args:
            hkl : Set of input Miller indices.

        Returns:
            A set of Miller indices.
        """
        rec_vec = hkl.dot(self.basis.mat)
        rec_abs = np.sqrt((rec_vec**2).sum(axis=-1))
        rec_th = np.arccos(-rec_vec[..., 2] / rec_abs)
        src_th = rec_th - np.arccos(0.5 * rec_abs)
        return np.abs(np.sin(src_th)) < np.arccos(self.setup.kin_max[2])

    def generate_streaks(self, hkl: np.ndarray, width: float, return_idxs: bool=False) -> Streaks:
        """Generate a CBD pattern. Return a set of streaks in :class:`cbclib.Streaks` container.

        Args:
            hkl : Set of Miller indices.
            width : Width of diffraction streaks in pixels.
            return_idxs : Return a set of input streak indices that satisfy the Bragg condition.

        Returns:
            A set of streaks, that constitute the predicted CBD pattern.
        """
        kin, mask = calc_source_lines(basis=self.basis.mat, hkl=hkl, kin_min=self.setup.kin_min,
                                      kin_max=self.setup.kin_max)
        idxs = np.arange(hkl.shape[0])[mask]
        rec_vec = hkl[idxs].dot(self.basis.mat)[:, None]

        x, y = self.sample.kout_to_detector(kin + rec_vec, self.setup, rec_vec)
        if self.transform:
            x, y = self.transform.forward_points(x, y)

        if self.shape:
            mask = (0 < y).any(axis=1) & (y < self.shape[0]).any(axis=1) & \
                   (0 < x).any(axis=1) & (x < self.shape[1]).any(axis=1)
            x, y, idxs = x[mask], y[mask], idxs[mask]
        streaks = Streaks(x0=x[:, 0], y0=y[:, 0], x1=x[:, 1], y1=y[:, 1],
                          h=hkl[idxs][:, 0], k=hkl[idxs][:, 1], l=hkl[idxs][:, 2],
                          width=width * np.ones(x.shape[0]))

        if return_idxs:
            return streaks, idxs
        return streaks

    def filter_streaks(self, hkl: np.ndarray, signal: np.ndarray, background: np.ndarray,
                       width: float,  threshold: float=0.95, profile: str='gauss',
                       num_threads: int=1) -> Streaks:
        """Generate a predicted pattern and filter out all the streaks, which signal-to-noise ratio
        is below the ``threshold``.

        Args:
            hkl : Set of reciprocal lattice point to use for prediction.
            signal : Measured signal.
            background : Measured background, standard deviation of the signal is the square root of
                background.
            width : Difrraction streak width in pixels of a predicted pattern.
            threshold : SNR threshold.
            profile : Line width profiles of generated streaks. The following keyword values are
                allowed:

                * `tophat` : Top-hat (rectangular) function profile.
                * `linear` : Linear (triangular) function profile.
                * `quad` : Quadratic (parabola) function profile.
                * `gauss` : Gaussian function profile.

            num_threads : Number of threads used in the calculations.

        Returns:
            A set of filtered out streaks, SNR of which is above the ``threshold``.
        """
        streaks = self.generate_streaks(hkl, width)
        pattern = streaks.pattern_dataframe(self.shape, profile=profile)
        pattern = pattern.drop_duplicates(['x', 'y'], keep=False).reset_index(drop=True)
        idxs = filter_hkl(sgn=signal, bgd=background, coord=pattern[['x', 'y']].to_numpy(),
                          prof=pattern['rp'].to_numpy(), idxs=pattern['index'].to_numpy(),
                          threshold=threshold, num_threads=num_threads)
        return streaks.mask_streaks(idxs)

    def pattern_dataframe(self, hkl: np.ndarray, width: float, profile: str='gauss') -> pd.DataFrame:
        """Predict a CBD pattern and return in the :class:`pandas.DataFrame` format.

        Args:
            hkl : Set of reciprocal lattice point to use for prediction.
            width : Difrraction streak width in pixels of a predicted pattern.
            profile : Line width profiles. The following keyword values are allowed:

                * `tophat` : Top-hat (rectangular) function profile.
                * `linear` : Linear (triangular) function profile.
                * `quad` : Quadratic (parabola) function profile.
                * `gauss` : Gaussian function profile.

        Returns:
            A pattern in :class:`pandas.DataFrame` format.
        """
        streaks = self.generate_streaks(hkl, width)
        return streaks.pattern_dataframe(self.shape, profile=profile)
