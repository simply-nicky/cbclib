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
from multiprocessing import cpu_count
from dataclasses import dataclass
from typing import (Any, ClassVar, Dict, Iterator, Sequence, List, Optional, Set, Tuple, Union,
                    get_type_hints)
import numpy as np
import pandas as pd
from .src import (tilt_matrix, euler_angles, det_to_k, k_to_det, k_to_smp, rotate, euler_matrix,
                  tilt_angles, draw_line_image, draw_line_mask, draw_line_table, source_lines,
                  unique_indices)
from .cxi_protocol import Indices
from .data_container import DataContainer, Parser, INIParser, JSONParser, Transform, Crop

@dataclass
class Basis(DataContainer):
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
    def parser(cls, ext: str='ini') -> Parser:
        if ext == 'ini':
            return INIParser({'basis': ('a_vec', 'b_vec', 'c_vec')},
                             types=get_type_hints(cls))
        if ext == 'json':
            return JSONParser({'basis': ('a_vec', 'b_vec', 'c_vec')})

        raise ValueError(f"Invalid format: {ext}")

    @classmethod
    def read(cls, file: str, ext: str='ini') -> Basis:
        return cls(**cls.parser(ext).read(file))

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
        return cls.import_matrix(np.stack((mat[:, 0] * np.sin(mat[:, 1]) * np.cos(mat[:, 2]),
                                           mat[:, 0] * np.sin(mat[:, 1]) * np.sin(mat[:, 2]),
                                           mat[:, 0] * np.cos(mat[:, 1])), axis=1))

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
        lengths = np.sqrt(np.sum(self.mat**2, axis=1))
        return np.stack((lengths, np.cos(self.mat[:, 2] / lengths),
                         np.arctan2(self.mat[:, 1], self.mat[:, 0])), axis=1)

@dataclass
class ScanSetup(DataContainer):
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

    @classmethod
    def parser(cls, ext: str='ini') -> Parser:
        if ext == 'ini':
            return INIParser({'exp_geom': ('foc_pos', 'pupil_roi', 'rot_axis', 'smp_dist',
                                           'wavelength', 'x_pixel_size', 'y_pixel_size')},
                             types=get_type_hints(cls))
        if ext == 'json':
            return JSONParser({'exp_geom': ('foc_pos', 'pupil_roi', 'rot_axis', 'smp_dist',
                                            'wavelength', 'x_pixel_size', 'y_pixel_size')})

        raise ValueError(f"Invalid format: {ext}")

    @classmethod
    def read(cls, file: str, ext: str='ini') -> ScanSetup:
        return cls(**cls.parser(ext).read(file))

    def detector_to_kout(self, x: np.ndarray, y: np.ndarray, pos: np.ndarray,
                         idxs: Optional[np.ndarray]=None, num_threads: int=1) -> np.ndarray:
        """Project detector coordinates ``(x, y)`` to the output wave-vectors space originating
        from the point ``pos``.

        Args:
            x : A set of x coordinates.
            y : A set of y coordinates.
            pos : Source point of the output wave-vectors.
            idxs : Source point indices.
            num_threads : Number of threads used in the calculations.

        Returns:
            An array of output wave-vectors.
        """
        x = np.atleast_1d(x * self.x_pixel_size)
        y = np.atleast_1d(y * self.y_pixel_size)
        return det_to_k(x, y, pos, idxs=idxs, num_threads=num_threads)

    def kout_to_detector(self, kout: np.ndarray, pos: np.ndarray, idxs: Optional[np.ndarray]=None,
                         num_threads: int=1) -> Tuple[np.ndarray, np.ndarray]:
        """Project output wave-vectors originating from the point ``pos`` to the detector plane.

        Args:
            kout : Output wave-vectors.
            pos : Source point of the output wave-vectors.
            idxs : Source point indices.
            num_threads : Number of threads used in the calculations.

        Returns:
            A tuple of x and y detector coordinates.
        """
        det_x, det_y = k_to_det(kout, pos, idxs=idxs, num_threads=num_threads)
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
        x = np.atleast_1d(x * self.x_pixel_size)
        y = np.atleast_1d(y * self.y_pixel_size)
        return det_to_k(x, y, self.foc_pos, num_threads=num_threads)

    def kin_to_detector(self, kin: np.ndarray, num_threads: int=1) -> Tuple[np.ndarray, np.ndarray]:
        """Project incident wave-vectors to the detector plane.

        Args:
            kin : Incident wave-vectors.
            num_threads : Number of threads used in the calculations.

        Returns:
            A tuple of x and y detector coordinates.
        """
        det_x, det_y = k_to_det(kin, self.foc_pos, num_threads=num_threads)
        return det_x / self.x_pixel_size, det_y / self.y_pixel_size

    def kin_to_sample(self, kin: np.ndarray, smp_z: Union[np.ndarray, float, None]=None,
                      idxs: Optional[np.ndarray]=None, num_threads: int=1) -> np.ndarray:
        """Project incident wave-vectors to the sample planes located at the z coordinates
        ``smp_z``.

        Args:
            kin : Incident wave-vectors.
            smp_z : z coordinates of the sample [m].
            idxs : Sample indices.
            num_threads : Number of threads used in the calculations.

        Returns:
            An array of points pertaining to the sample planes.
        """
        if smp_z is None:
            smp_z = self.foc_pos[2] + self.smp_dist
        return k_to_smp(kin, np.atleast_1d(smp_z), self.foc_pos, idxs=idxs,
                        num_threads=num_threads)

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
class Sample(DataContainer):
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

    def kin_to_sample(self, setup: ScanSetup, kin: Optional[np.ndarray]=None,
                      num_threads: int=1) -> np.ndarray:
        """Project incident wave-vectors ``kin`` to the sample plane.

        Args:
            setup : Experimental setup.
            kin : Incident wave-vectors.
            num_threads : Number of threads used in the calculations.

        Returns:
            An array of points belonging to the sample plane.
        """
        kin = setup.kin_center if kin is None else kin
        return setup.kin_to_sample(kin, self.z, num_threads=num_threads)

    def rotate_basis(self, basis: Basis) -> Basis:
        """Rotate a :class:`cbclib.Basis` by the ``rotation`` attribute.

        Args:
            basis : Indexing solution basis vectors.

        Returns:
            A new rotated :class:`cbclib.Basis` object.
        """
        return Basis.import_matrix(self.rotation(basis.mat))

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
        kout = setup.detector_to_kout(x, y, self.kin_to_sample(setup), num_threads=num_threads)
        if rec_vec is not None:
            smp_pos = self.kin_to_sample(setup, kout - rec_vec, num_threads=num_threads)
            idxs = np.arange(x.size, dtype=np.uint32)
            kout = setup.detector_to_kout(x, y, smp_pos, idxs, num_threads=num_threads)
        return kout

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
        x, y = setup.kout_to_detector(kout, self.kin_to_sample(setup), num_threads=num_threads)
        if rec_vec is not None:
            smp_pos = self.kin_to_sample(setup, kout - rec_vec, num_threads=num_threads)
            idxs = np.arange(smp_pos.size / smp_pos.shape[-1], dtype=np.uint32)
            x, y = setup.kout_to_detector(kout, smp_pos, idxs, num_threads=num_threads)
        return x, y

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

@dataclass
class ScanSamples():
    """A collection of sample :class:`cbclib.Sample` objects. Provides an interface to import
    from and exprort to a :class:`pandas.DataFrame` table and a set of dictionary methods.
    """
    samples : Dict[int, Sample]

    def __getitem__(self, idxs: Indices) -> ScanSamples:
        return ScanSamples(self.samples[idxs])

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

    @property
    def z(self) -> np.ndarray:
        return np.array([sample.z for sample in self.samples.values()])

    @property
    def rmats(self) -> np.ndarray:
        return np.stack([sample.rotation.matrix for sample in self.samples.values()])

    def kin_to_sample(self, setup: ScanSetup, kin: Optional[np.ndarray]=None,
                      idxs: Optional[np.ndarray]=None, num_threads: int=1) -> np.ndarray:
        """Project incident wave-vectors to the sample planes.

        Args:
            setup : Experimental setup.
            kin : An array of incident wave-vectors.
            idxs : Sample indices.
            num_threads : Number of threads used in the calculations.

        Returns:
            Array of sample coordinates.
        """
        if kin is None:
            kin = np.tile(setup.kin_center[None], (len(self.samples), 1))
            idxs = np.arange(len(self.samples), dtype=np.uint32)
        return setup.kin_to_sample(kin, self.z, idxs, num_threads=num_threads)

    def detector_to_kout(self, x: np.ndarray, y: np.ndarray, setup: ScanSetup, idxs: np.ndarray,
                         rec_vec: Optional[np.ndarray]=None, num_threads: int=1) -> np.ndarray:
        """Project detector coordinates ``(x, y)`` to the output wave-vectors space originating
        from the samples' locations.

        Args:
            x : A set of x coordinates.
            y : A set of y coordinates.
            setup : Experimental setup.
            idxs : Sample indices.
            rec_vec : A set of scattering vectors corresponding to the detector points.
            num_threads : Number of threads used in the calculations.

        Returns:
            An array of output wave-vectors.
        """
        kout = setup.detector_to_kout(x, y, self.kin_to_sample(setup), idxs, num_threads)
        if rec_vec is not None:
            smp_pos = self.kin_to_sample(setup, kout - rec_vec, idxs, num_threads)
            idxs = np.arange(x.size, dtype=np.uint32)
            kout = setup.detector_to_kout(x, y, smp_pos, idxs, num_threads)
        return kout

    def kout_to_detector(self, kout: np.ndarray, setup: ScanSetup,
                         idxs: np.ndarray, rec_vec: Optional[np.ndarray]=None,
                         num_threads: int=1) -> Tuple[np.ndarray, np.ndarray]:
        """Project output wave-vectors originating from the samples' locations to the detector
        plane.

        Args:
            kout : Output wave-vectors.
            setup : Experimental setup.
            idxs : Sample indices.
            rec_vec : A set of scattering vectors corresponding to the output wave-vectors.
            num_threads : Number of threads used in the calculations.

        Returns:
            A tuple of x and y detector coordinates.
        """
        x, y = setup.kout_to_detector(kout, self.kin_to_sample(setup), idxs, num_threads)
        if rec_vec is not None:
            smp_pos = self.kin_to_sample(setup, kout - rec_vec, idxs, num_threads)
            idxs = np.arange(smp_pos.size / smp_pos.shape[-1], dtype=np.uint32)
            x, y = setup.kout_to_detector(kout, smp_pos, idxs, num_threads)
        return x, y

    def rotate(self, vectors: np.ndarray, idxs: np.ndarray, reciprocate: bool=False,
               num_threads: int=1) -> np.ndarray:
        """Rotate an array of vectors into the samples' system of coordinates.

        Args:
            vectors : An array of vectors.
            idxs : Sample indices.
            reciprocate : Apply the inverse sample rotations if True.
            num_threads : Number of threads used in the calculations.

        Returns:
            An array of rotated vectors.
        """
        rmats = np.swapaxes(self.rmats, 1, 2) if reciprocate else self.rmats
        return rotate(vectors, rmats, idxs, num_threads)

    def to_dataframe(self) -> pd.DataFrame:
        """Export the sample object to a :class:`pandas.DataFrame` table.

        Returns:
            A :class:`pandas.DataFrame` table with the following columns:

            * `Rxx`, `Rxy`, `Rxz`, `Ryx`, `Ryy`, `Ryz`, `Rzx`, `Rzy`, `Rzz` : Rotational
              matrix.
            * `z` : z coordinate [m].
        """
        return pd.DataFrame((sample.to_dataframe() for sample in self.samples.values()), index=self.samples.keys())

class StreaksAbc:
    def pattern_dataframe(self, width: float, shape: Optional[Tuple[int, int]]=None,
                          kernel: str='rectangular', reduce: bool=True) -> pd.DataFrame:
        """Draw a pattern in the :class:`pandas.DataFrame` format.

        Args:
            width : Lines width in pixels.
            shape : Detector grid shape.
            kernel : Choose one of the supported kernel functions [Krn]_. The following kernels
                are available:

                * 'biweigth' : Quartic (biweight) kernel.
                * 'gaussian' : Gaussian kernel.
                * 'parabolic' : Epanechnikov (parabolic) kernel.
                * 'rectangular' : Uniform (rectangular) kernel.
                * 'triangular' : Triangular kernel.

            reduce : Discard the pixel data with reflection profile values equal to
                zero.

        Returns:
            A pattern in :class:`pandas.DataFrame` format.
        """
        df = pd.DataFrame(self.pattern_dict(width=width, shape=shape, kernel=kernel))
        if reduce:
            return df[df['rp'] > 0.0]
        return df

    def pattern_dict(self, width: float, shape: Optional[Tuple[int, int]]=None,
                     kernel: str='rectangular') -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def pattern_image(self, width: float, shape: Tuple[int, int],
                      kernel: str='gaussian') -> np.ndarray:
        """Draw a pattern in the :class:`numpy.ndarray` format.

        Args:
            width : Lines width in pixels.
            shape : Detector grid shape.
            kernel : Choose one of the supported kernel functions [Krn]_. The following kernels
                are available:

                * 'biweigth' : Quartic (biweight) kernel.
                * 'gaussian' : Gaussian kernel.
                * 'parabolic' : Epanechnikov (parabolic) kernel.
                * 'rectangular' : Uniform (rectangular) kernel.
                * 'triangular' : Triangular kernel.

        Returns:
            A pattern in :class:`numpy.ndarray` format.
        """
        return draw_line_image(self.to_lines(width), shape=shape, kernel=kernel)

    def pattern_mask(self, width: float, shape: Tuple[int, int], max_val: int=1,
                     kernel: str='rectangular') -> np.ndarray:
        """Draw a pattern mask.

        Args:
            width : Lines width in pixels.
            shape : Detector grid shape.
            max_val : Mask maximal value.
            kernel : Choose one of the supported kernel functions [Krn]_. The following kernels
                are available:

                * 'biweigth' : Quartic (biweight) kernel.
                * 'gaussian' : Gaussian kernel.
                * 'parabolic' : Epanechnikov (parabolic) kernel.
                * 'rectangular' : Uniform (rectangular) kernel.
                * 'triangular' : Triangular kernel.

        Returns:
            A pattern mask.
        """
        return draw_line_mask(self.to_lines(width), shape=shape, max_val=max_val,
                              kernel=kernel)

    def to_lines(self, width: float) -> np.ndarray:
        raise NotImplementedError

@dataclass
class Streaks(StreaksAbc, DataContainer):
    """Detector streak lines container. Provides an interface to draw a pattern for a set of
    lines.

    Args:
        x0 : x coordinates of the first point of a line.
        y0 : y coordinates of the first point of a line.
        x1 : x coordinates of the second point of a line.
        y1 : y coordinates of the second point of a line.
        length: Line's length in pixels.
        h : First Miller index.
        k : Second Miller index.
        l : Third Miller index.
        hkl_id : Bragg reflection index.
    """
    columns     : ClassVar[Set[str]] = {'x0', 'y0', 'x1', 'y1', 'length', 'h', 'k', 'l', 'hkl_id'}

    x0          : np.ndarray
    y0          : np.ndarray
    x1          : np.ndarray
    y1          : np.ndarray
    length      : Optional[np.ndarray] = None
    h           : Optional[np.ndarray] = None
    k           : Optional[np.ndarray] = None
    l           : Optional[np.ndarray] = None
    hkl_id      : Optional[np.ndarray] = None

    def __post_init__(self):
        if self.length is None:
            self.length = np.sqrt((self.x1 - self.x0)**2 + (self.y1 - self.y0)**2)

    @classmethod
    def import_dataframe(cls, data: pd.DataFrame) -> Streaks:
        return cls(**{key: data[key] for key in cls.columns.intersection(data.columns)})

    @property
    def hkl(self) -> Optional[np.ndarray]:
        if self.h is None or self.k is None or self.l is None:
            return None
        return np.stack((self.h, self.k, self.l), axis=1)


    def __len__(self) -> int:
        return self.length.shape[0]

    def keys(self) -> List[str]:
        keys = ['index', 'x' , 'y', 'rp']
        if self.hkl is not None:
            keys.extend(['h', 'k', 'l'])
        if self.hkl_id is not None:
            keys.append('hkl_id')
        return keys

    def mask_streaks(self, idxs: Indices) -> Streaks:
        """Return a new streaks container with a set of streaks discarded.

        Args:
            idxs : A set of indices of the streaks to discard.

        Returns:
            A new :class:`cbclib.Streaks` container.
        """
        return Streaks(**{attr: self[attr][idxs] for attr in self.contents()})

    def pattern_dict(self, width: float, shape: Optional[Tuple[int, int]]=None,
                     kernel: str='rectangular') -> Dict[str, np.ndarray]:
        """Draw a pattern in the :class:`dict` format.

        Args:
            width : Lines width in pixels.
            shape : Detector grid shape.
            kernel : Choose one of the supported kernel functions [Krn]_. The following kernels
                are available:

                * 'biweigth' : Quartic (biweight) kernel.
                * 'gaussian' : Gaussian kernel.
                * 'parabolic' : Epanechnikov (parabolic) kernel.
                * 'rectangular' : Uniform (rectangular) kernel.
                * 'triangular' : Triangular kernel.

        Returns:
            A pattern in dictionary format.
        """
        idx, x, y, rp = draw_line_table(lines=self.to_lines(width), shape=shape,
                                        kernel=kernel)
        pattern = {'index': idx, 'x': x, 'y': y, 'rp': rp}
        if self.hkl is not None:
            pattern.update(h=self.h[idx], k=self.k[idx], l=self.l[idx])
        if self.hkl_id is not None:
            pattern['hkl_id'] = self.hkl_id[idx]
        return pattern

    def to_dataframe(self) -> pd.DataFrame:
        """Export a streaks container into :class:`pandas.DataFrame`.

        Returns:
            A dataframe with all the data specified in :class:`cbclib.Streaks`.
        """
        return pd.DataFrame({attr: self[attr] for attr in self.contents()})

    def to_lines(self, width: float) -> np.ndarray:
        """Export a streaks container into line parameters ``x0, y0, x1, y1, width``:

        * `[x0, y0]`, `[x1, y1]` : The coordinates of the line's ends.
        * `width` : Line's width.

        Returns:
            An array of line parameters.
        """
        widths = width * np.ones(len(self))
        return np.stack((self.x0, self.y0, self.x1, self.y1, widths), axis=1)

@dataclass
class ScanStreaks(StreaksAbc):
    columns : ClassVar[Set[str]] = Streaks.columns
    streaks : Dict[int, Streaks]

    def __getitem__(self, idxs: Indices) -> ScanStreaks:
        return ScanStreaks(self.streaks[idxs])

    @classmethod
    def import_dataframe(cls, df: pd.DataFrame) -> ScanStreaks:
        frames, fidxs, _ = unique_indices(df['frames'].to_numpy())
        return cls({frames[index]: Streaks.import_dataframe(df[fidxs[index]:fidxs[index + 1]])
                    for index in range(frames.size)})

    @classmethod
    def import_dict(cls, dct: Dict[str, np.ndarray]) -> ScanStreaks:
        frames, fidxs, _ = unique_indices(dct['frames'])

        result = {}
        for index in range(frames.size):
            result[frames[index]] = Streaks(**{key: dct[key][fidxs[index]:fidxs[index + 1]]
                                               for key in cls.columns.intersection(dct.keys())})

        return cls(result)

    def keys(self) -> List[str]:
        return self.streaks[0].keys()

    def pattern_dict(self, width: float, shape: Optional[Tuple[int, int]]=None,
                     kernel: str='rectangular') -> Dict[str, np.ndarray]:
        n_streaks = 0
        result = {key: [] for key in self.keys()}

        for idx, streaks in self.streaks.items():
            dct = streaks.pattern_dict(width, shape, kernel)

            dct['index'] += n_streaks
            dct['frames'] = idx * np.ones(len(streaks), dtype=int)
            n_streaks += len(streaks)

            for attr, val in dct.items():
                result[attr].append(val)

        for attr in result:
            result[attr] = np.concatenate(result[attr])

        return result

    def to_lines(self, width: float) -> Dict[int, np.ndarray]:
        return {idx: val.to_lines(width) for idx, val in self.streaks.items()}

@dataclass
class CBDModel(DataContainer):
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
    samples     : ScanSamples
    setup       : ScanSetup
    transform   : Optional[Transform] = None
    shape       : Optional[Tuple[int, int]] = None
    num_threads : int = cpu_count()

    def __post_init__(self):
        if isinstance(self.transform, Crop):
            self.shape = (self.transform.roi[1] - self.transform.roi[0],
                          self.transform.roi[3] - self.transform.roi[2])

    def __getitem__(self, idxs: Indices) -> CBDModel:
        return self.replace(samples=self.samples[idxs])

    def bases(self) -> Iterator[Basis]:
        for sample in self.samples.samples.values():
            yield sample.rotate_basis(self.basis)

    def filter_hkl(self, hkl: np.ndarray) -> np.ndarray:
        """Return a set of reciprocal lattice points that lie in the region of reciprocal space
        involved in diffraction.

        Args:
            hkl : Set of input Miller indices.

        Returns:
            A set of Miller indices.
        """
        mask = np.ones(hkl.shape[0], dtype=bool)
        for basis in self.bases():
            rec_vec = hkl.dot(basis.mat)
            rec_abs = np.sqrt((rec_vec**2).sum(axis=-1))
            rec_th = np.arccos(-rec_vec[..., 2] / rec_abs)
            src_th = rec_th - np.arccos(0.5 * rec_abs)
            mask &= np.abs(np.sin(src_th)) < np.arccos(self.setup.kin_max[2])
        return mask

    def generate_streaks(self, hkl: np.ndarray, hkl_index: bool=False) -> ScanStreaks:
        """Generate a CBD pattern. Return a set of streaks in :class:`cbclib.Streaks` container.

        Args:
            hkl : Set of Miller indices.
            width : Width of diffraction streaks in pixels.
            hkl_index : Save ``hkl`` indices in the streaks container if True.

        Returns:
            A set of streaks, that constitute the predicted CBD pattern.
        """
        bases = np.array([basis.mat for basis in self.bases()])
        kin, mask = source_lines(basis=bases, hkl=hkl, kin_min=self.setup.kin_min,
                                 kin_max=self.setup.kin_max, num_threads=self.num_threads)

        frames = np.repeat(np.arange(bases.shape[0]), hkl.shape[0])[mask]
        idxs = np.tile(np.arange(hkl.shape[0]), bases.shape[0])[mask]
        rec_vec = np.tensordot(hkl, bases, axes=(-1, -2)).reshape(-1, 3)[mask]

        x, y = self.samples.kout_to_detector(kin + rec_vec, self.setup, idxs=frames,
                                             rec_vec=rec_vec, num_threads=self.num_threads)
        if self.transform:
            x, y = self.transform.forward_points(x, y)

        if self.shape:
            mask = (0 < y).any(axis=1) & (y < self.shape[0]).any(axis=1) & \
                   (0 < x).any(axis=1) & (x < self.shape[1]).any(axis=1)
            x, y, frames, idxs = x[mask], y[mask], frames[mask], idxs[mask]

        result = {'frames': frames, 'x0': x[:, 0], 'y0': y[:, 0], 'x1': x[:, 1], 'y1': y[:, 1],
                  'h': hkl[idxs][:, 0], 'k': hkl[idxs][:, 1], 'l': hkl[idxs][:, 2]}
        if hkl_index:
            result['hkl_id'] = idxs

        return ScanStreaks.import_dict(result)
