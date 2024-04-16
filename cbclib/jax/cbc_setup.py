from __future__ import annotations

from multiprocessing import cpu_count
from typing import Any, ClassVar, Iterable, List, Optional, Tuple, Union, cast, get_type_hints
import pandas as pd
import numpy as np
from numpy.linalg import eigh
import jax.numpy as jnp
from .dataclasses import jax_dataclass, field
from .primitives import (det_to_k, euler_angles, euler_matrix, k_to_det, k_to_smp, matmul,
                         source_lines, tilt_angles, tilt_matrix)
from ..src import draw_line_image, draw_line_mask, draw_line_table
from ..data_container import DataContainer, Parser, INIParser, JSONParser, Transform, Crop
from ..annotations import (Indices, JaxArray, JaxIntArray, JaxRealArray, NDIntArray, NDRealArray,
                           Pattern, PatternWithHKL, PatternWithHKLID, RealSequence, Shape)

@jax_dataclass
class Basis(DataContainer):
    """An indexing solution, defined by a set of three unit cell vectors.

    Args:
        a_vec : First basis vector.
        b_vec : Second basis vector.
        c_vec : Third basis vector.
    """
    a_vec : JaxRealArray
    b_vec : JaxRealArray
    c_vec : JaxRealArray

    def __post_init__(self):
        self.mat = jnp.stack((self.a_vec, self.b_vec, self.c_vec))

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
    def import_matrix(cls, mat: JaxRealArray) -> Basis:
        """Return a new :class:`Basis` object, initialised by a stacked matrix of three basis
        vectors.

        Args:
            mat : A matrix of three stacked basis vectors.

        Returns:
            A new :class:`Basis` object.
        """
        if mat.size != 9:
            raise ValueError("Wrong matrix size: " + str(mat.size))
        mat = mat.reshape(-1, 3)
        return cls(a_vec=mat[0], b_vec=mat[1], c_vec=mat[2])

    @classmethod
    def import_spherical(cls, mat: JaxRealArray) -> Basis:
        """Return a new :class:`Basis` object, initialised by a stacked matrix of three basis
        vectors written in spherical coordinate system.

        Args:
            mat : A matrix of three stacked basis vectors in spherical coordinate system.

        Returns:
            A new :class:`Basis` object.
        """
        return cls.import_matrix(jnp.stack((mat[:, 0] * jnp.sin(mat[:, 1]) * jnp.cos(mat[:, 2]),
                                            mat[:, 0] * jnp.sin(mat[:, 1]) * jnp.sin(mat[:, 2]),
                                            mat[:, 0] * jnp.cos(mat[:, 1])), axis=1))

    def generate_hkl(self, q_abs: float) -> JaxIntArray:
        """Return a set of reflections lying inside of a sphere of radius ``q_abs`` in the
        reciprocal space.

        Args:
            q_abs : The radius of a sphere in the reciprocal space.

        Returns:
            An array of Miller indices, that lie inside of the sphere.
        """
        lat_size = jnp.rint(q_abs / self.to_spherical()[:, 0]).astype(int)
        h_idxs = jnp.arange(-lat_size[0], lat_size[0] + 1)
        k_idxs = jnp.arange(-lat_size[1], lat_size[1] + 1)
        l_idxs = jnp.arange(-lat_size[2], lat_size[2] + 1)
        h_grid, k_grid, l_grid = jnp.meshgrid(h_idxs, k_idxs, l_idxs)
        hkl = jnp.stack((h_grid.ravel(), k_grid.ravel(), l_grid.ravel()), axis=1)
        hkl = jnp.compress(jnp.any(hkl, axis=1), hkl, axis=0)

        rec_vec = jnp.dot(hkl, self.mat)
        rec_abs = jnp.sqrt(jnp.sum(rec_vec**2, axis=-1))
        return hkl[rec_abs < q_abs]

    def lattice_constants(self) -> JaxRealArray:
        r"""Return lattice constants :math:`a, b, c, \alpha, \beta, \gamma`. The unit cell
        length are unitless.

        Returns:
            An array of lattice constants.
        """
        lengths = self.to_spherical()[:, 0]
        alpha = jnp.arccos(jnp.sum(self.mat[1] * self.mat[2]) / (lengths[1] * lengths[2]))
        beta = jnp.arccos(jnp.sum(self.mat[0] * self.mat[2]) / (lengths[0] * lengths[2]))
        gamma = jnp.arccos(jnp.sum(self.mat[0] * self.mat[1]) / (lengths[0] * lengths[1]))
        return jnp.concatenate((lengths, [alpha, beta, gamma]))

    def reciprocate(self) -> Basis:
        """Calculate the basis of the reciprocal lattice.

        Returns:
            The basis of the reciprocal lattice.
        """
        a_rec = jnp.cross(self.b_vec, self.c_vec) / jnp.dot(jnp.cross(self.b_vec, self.c_vec),
                                                            self.a_vec)
        b_rec = jnp.cross(self.c_vec, self.a_vec) / jnp.dot(jnp.cross(self.c_vec, self.a_vec),
                                                            self.b_vec)
        c_rec = jnp.cross(self.a_vec, self.b_vec) / jnp.dot(jnp.cross(self.a_vec, self.b_vec),
                                                            self.c_vec)
        return Basis.import_matrix(jnp.stack((a_rec, b_rec, c_rec)))

    def to_spherical(self) -> JaxRealArray:
        """Return a stack of unit cell vectors in spherical coordinate system.

        Returns:
            A matrix of three stacked unit cell vectors in spherical coordinate system.
        """
        lengths = jnp.sqrt(jnp.sum(self.mat**2, axis=1))
        return jnp.stack((lengths, jnp.cos(self.mat[:, 2] / lengths),
                          jnp.arctan2(self.mat[:, 1], self.mat[:, 0])), axis=1)

@jax_dataclass
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
    foc_pos         : JaxRealArray
    pupil_roi       : JaxRealArray
    rot_axis        : JaxRealArray
    smp_dist        : float
    wavelength      : float = field(static=True)
    x_pixel_size    : float = field(static=True)
    y_pixel_size    : float = field(static=True)

    @property
    def kin_min(self) -> JaxRealArray:
        return self.detector_to_kin(x=self.pupil_roi[2], y=self.pupil_roi[0]).ravel()

    @property
    def kin_max(self) -> JaxRealArray:
        return self.detector_to_kin(x=self.pupil_roi[3], y=self.pupil_roi[1]).ravel()

    @property
    def kin_center(self) -> JaxRealArray:
        return self.detector_to_kin(x=jnp.mean(self.pupil_roi[2:]),
                                    y=jnp.mean(self.pupil_roi[:2])).ravel()

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

    def detector_to_kout(self, x: RealSequence, y: RealSequence, pos: JaxRealArray,
                         idxs: Optional[JaxIntArray]=None, num_threads: int=1) -> JaxRealArray:
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
        return det_to_k(jnp.atleast_1d(x) * self.x_pixel_size,
                        jnp.atleast_1d(y) * self.y_pixel_size,
                        pos, idxs=idxs, num_threads=num_threads)

    def kout_to_detector(self, kout: JaxRealArray, pos: JaxRealArray,
                         idxs: Optional[JaxIntArray]=None,
                         num_threads: int=1) -> Tuple[JaxRealArray, JaxRealArray]:
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

    def detector_to_kin(self, x: RealSequence, y: RealSequence, num_threads: int=1) -> JaxRealArray:
        """Project detector coordinates ``(x, y)`` to the incident wave-vectors space.

        Args:
            x : A set of x coordinates.
            y : A set of y coordinates.
            num_threads : Number of threads used in the calculations.

        Returns:
            An array of incident wave-vectors.
        """
        return det_to_k(jnp.atleast_1d(x) * self.x_pixel_size,
                        jnp.atleast_1d(y) * self.y_pixel_size,
                        self.foc_pos, num_threads=num_threads)

    def kin_to_detector(self, kin: JaxRealArray, num_threads: int=1) -> Tuple[JaxRealArray,
                                                                              JaxRealArray]:
        """Project incident wave-vectors to the detector plane.

        Args:
            kin : Incident wave-vectors.
            num_threads : Number of threads used in the calculations.

        Returns:
            A tuple of x and y detector coordinates.
        """
        det_x, det_y = k_to_det(kin, self.foc_pos, num_threads=num_threads)
        return det_x / self.x_pixel_size, det_y / self.y_pixel_size

    def kin_to_sample(self, kin: JaxRealArray, smp_z: Optional[RealSequence]=None,
                      idxs: Optional[JaxIntArray]=None, num_threads: int=1) -> JaxRealArray:
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
        return k_to_smp(kin, jnp.atleast_1d(smp_z), self.foc_pos, idxs=idxs,
                        num_threads=num_threads)

    def tilt_rotation(self, theta: float) -> Rotation:
        """Return a tilt rotation by the angle ``theta`` arount the axis of rotation.

        Args:
            theta : Angle of rotation.

        Returns:
            A new :class:`cbclib.Rotation` object.
        """
        return Rotation.import_tilt(jnp.asarray([theta, self.rot_axis[0], self.rot_axis[1]]))

    def tilt_samples(self, frames: JaxIntArray, thetas: JaxRealArray) -> ScanSamples:
        """Return a list of sample position and orientations of a tilt series.

        Args:
            frames : Set of frame indices.
            thetas : Set of sample tilts.

        Returns:
            A container of sample objects :class:`ScanSamples`.
        """
        angles = jnp.empty((thetas.size, 3))
        angles[:, 0] = thetas
        angles[:, 1:] = self.rot_axis
        rmats = tilt_matrix(angles).reshape(-1, 3, 3)
        return ScanSamples(frames, rmats, jnp.full(frames.size, self.foc_pos[2] + self.smp_dist))

@jax_dataclass
class Rotation(DataContainer):
    """A rotation matrix implementation. Provides auxiliary methods to work with Euler
    and tilt angles.

    Args:
        matrix : Rotation matrix.
    """
    matrix : JaxRealArray = field(default=jnp.eye(3, 3))

    def __post_init__(self):
        self.matrix = self.matrix.reshape((3, 3))

    @classmethod
    def import_euler(cls, angles: RealSequence) -> Rotation:
        r"""Calculate a rotation matrix from Euler angles with Bunge convention [EUL]_.

        Args:
            angles : Euler angles :math:`\phi_1, \Phi, \phi_2`.

        Returns:
            A new rotation matrix :class:`Rotation`.
        """
        return cls(euler_matrix(jnp.asarray(angles)))

    @classmethod
    def import_tilt(cls, angles: RealSequence) -> Rotation:
        r"""Calculate a rotation matrix for a set of three angles set of three angles
        :math:`\theta, \alpha, \beta`, a rotation angle :math:`\theta`, an angle between the
        axis of rotation and OZ axis :math:`\alpha`, and a polar angle of the axis of rotation
        :math:`\beta`.

        Args:
            angles : A set of angles :math:`\theta, \alpha, \beta`.

        Returns:
            A new rotation matrix :class:`Rotation`.
        """
        return cls(tilt_matrix(jnp.asarray(angles)))

    def __call__(self, inp: JaxArray) -> JaxArray:
        """Apply the rotation to a set of vectors ``inp``.

        Args:
            inp : A set of 3D vectors.

        Returns:
            A set of rotated 3D vectors.
        """
        return jnp.dot(inp, self.matrix.T)

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

    def to_euler(self) -> JaxRealArray:
        r"""Calculate Euler angles with Bunge convention [EUL]_.

        Returns:
            A set of Euler angles with Bunge convention :math:`\phi_1, \Phi, \phi_2`.
        """
        return euler_angles(self.matrix)

    def to_tilt(self) -> JaxRealArray:
        r"""Calculate an axis of rotation and a rotation angle for a rotation matrix.

        Returns:
            A set of three angles :math:`\theta, \alpha, \beta`, a rotation angle :math:`\theta`,
            an angle between the axis of rotation and OZ axis :math:`\alpha`, and a polar angle
            of the axis of rotation :math:`\beta`.
        """
        if jnp.allclose(self.matrix, self.matrix.T):
            eigw, eigv = jnp.stack(eigh(self.matrix))
            axis = eigv[jnp.isclose(eigw, 1.0)]
            theta = jnp.arccos(0.5 * (jnp.trace(self.matrix) - 1.0))
            return jnp.array([theta, jnp.arccos(axis[0, 2]), jnp.arctan2(axis[0, 1], axis[0, 0])])
        return tilt_angles(self.matrix)

@jax_dataclass
class Sample(DataContainer):
    """A convergent beam sample implementation. Stores position and orientation of the sample.

    Args:
        rotation : rotation matrix, that defines the orientation of the sample.
        position : Sample's position [m].
    """
    rotation : Rotation
    z : JaxArray
    mat_columns : ClassVar[Tuple[str, ...]] = ('Rxx', 'Rxy', 'Rxz',
                                               'Ryx', 'Ryy', 'Ryz',
                                               'Rzx', 'Rzy', 'Rzz')
    z_column : ClassVar[str] = 'z'

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
        return cls(rotation=Rotation(jnp.asarray(data[list(cls.mat_columns)].to_numpy())),
                   z=data[cls.z_column])

    def kin_to_sample(self, setup: ScanSetup, kin: Optional[JaxRealArray]=None,
                      num_threads: int=1) -> JaxRealArray:
        """Project incident wave-vectors ``kin`` to the sample plane.

        Args:
            setup : Experimental setup.
            kin : Incident wave-vectors.
            num_threads : Number of threads used in the calculations.

        Returns:
            An array of points belonging to the sample plane.
        """
        if kin is None:
            kin = setup.kin_center
        return setup.kin_to_sample(kin, self.z, num_threads=num_threads)

    def rotate_basis(self, basis: Basis) -> Basis:
        """Rotate a :class:`cbclib.Basis` by the ``rotation`` attribute.

        Args:
            basis : Indexing solution basis vectors.

        Returns:
            A new rotated :class:`cbclib.Basis` object.
        """
        return Basis.import_matrix(self.rotation(basis.mat))

    def detector_to_kout(self, x: RealSequence, y: RealSequence, setup: ScanSetup,
                         rec_vec: Optional[JaxIntArray]=None,
                         num_threads: int=1) -> JaxRealArray:
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
            idxs = jnp.arange(jnp.asarray(x).size, dtype=jnp.uint32)
            kout = setup.detector_to_kout(x, y, smp_pos, idxs, num_threads=num_threads)
        return kout

    def kout_to_detector(self, kout: JaxRealArray, setup: ScanSetup,
                         rec_vec: Optional[JaxIntArray]=None,
                         num_threads: int=1) -> Tuple[JaxRealArray, JaxRealArray]:
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
            idxs = jnp.arange(smp_pos.size / smp_pos.shape[-1], dtype=jnp.uint32)
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
        return pd.Series(jnp.append(self.rotation.matrix.ravel(), self.z),
                         index=self.mat_columns + (self.z_column,))

@jax_dataclass
class ScanSamples():
    """A collection of sample :class:`cbclib.Sample` objects. Provides an interface to import
    from and exprort to a :class:`pandas.DataFrame` table and a set of dictionary methods.
    """
    frames : JaxIntArray = field(static=True)
    rmats : JaxRealArray
    zs : JaxRealArray

    def __getitem__(self, idxs: Indices) -> ScanSamples:
        return ScanSamples(self.frames[idxs], self.rmats[idxs], self.zs[idxs])

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
        samples = [Sample.import_dataframe(series) for _, series in df.iterrows()]
        return cls(jnp.asarray(frames), jnp.stack([sample.rotation.matrix for sample in samples]),
                   jnp.array([sample.z for sample in samples]))

    @property
    def size(self) -> int:
        return self.frames.size

    def samples(self) -> Iterable[Sample]:
        for rmat, z in zip(self.rmats, self.zs):
            yield Sample(Rotation(rmat), z)

    def kin_to_sample(self, setup: ScanSetup, kin: Optional[JaxRealArray]=None,
                      idxs: Optional[JaxIntArray]=None, num_threads: int=1) -> JaxRealArray:
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
            kin = cast(JaxRealArray, jnp.tile(setup.kin_center[None], (self.size, 1)))
        return setup.kin_to_sample(kin, self.zs, idxs, num_threads=num_threads)

    def detector_to_kout(self, x: RealSequence, y: RealSequence, setup: ScanSetup,
                         idxs: JaxIntArray, rec_vec: Optional[JaxRealArray]=None,
                         num_threads: int=1) -> JaxRealArray:
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
            idxs = jnp.arange(jnp.asarray(x).size, dtype=jnp.uint32)
            kout = setup.detector_to_kout(x, y, smp_pos, idxs, num_threads)
        return kout

    def kout_to_detector(self, kout: JaxRealArray, setup: ScanSetup,
                         idxs: JaxIntArray, rec_vec: Optional[JaxRealArray]=None,
                         num_threads: int=1) -> Tuple[JaxRealArray, JaxRealArray]:
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
            idxs = jnp.arange(smp_pos.size / smp_pos.shape[-1], dtype=jnp.uint32)
            x, y = setup.kout_to_detector(kout, smp_pos, idxs, num_threads)
        return x, y

    def rotate(self, vectors: JaxRealArray, idxs: JaxIntArray, reciprocate: bool=False,
               num_threads: int=1) -> JaxRealArray:
        """Rotate an array of vectors into the samples' system of coordinates.

        Args:
            vectors : An array of vectors.
            idxs : Sample indices.
            reciprocate : Apply the inverse sample rotations if True.
            num_threads : Number of threads used in the calculations.

        Returns:
            An array of rotated vectors.
        """
        rmats = jnp.swapaxes(self.rmats, 1, 2) if reciprocate else self.rmats
        return matmul(vectors, rmats, midxs=idxs, num_threads=num_threads)

    def to_dataframe(self) -> pd.DataFrame:
        """Export the sample object to a :class:`pandas.DataFrame` table.

        Returns:
            A :class:`pandas.DataFrame` table with the following columns:

            * `Rxx`, `Rxy`, `Rxz`, `Ryx`, `Ryy`, `Ryz`, `Rzx`, `Rzy`, `Rzz` : Rotational
              matrix.
            * `z` : z coordinate [m].
        """
        return pd.DataFrame((sample.to_dataframe() for sample in self.samples()),
                            index=np.array(self.frames))

@jax_dataclass
class Streaks(DataContainer):
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
    x0          : JaxRealArray
    y0          : JaxRealArray
    x1          : JaxRealArray
    y1          : JaxRealArray
    idxs        : JaxIntArray = field(default_factory=lambda: jnp.array([], dtype=int))
    length      : JaxRealArray = field(default_factory=lambda: jnp.array([]))
    h           : Optional[JaxIntArray] = field(default=None)
    k           : Optional[JaxIntArray] = field(default=None)
    l           : Optional[JaxIntArray] = field(default=None)
    hkl_id      : Optional[JaxIntArray] = field(default=None)

    def __post_init__(self):
        if self.idxs.shape != self.x0.shape:
            self.idxs = jnp.zeros(self.x0.shape, dtype=int)
        if self.length.shape != self.x0.shape:
            self.length = jnp.sqrt((self.x1 - self.x0)**2 + (self.y1 - self.y0)**2)

    @property
    def hkl(self) -> Optional[JaxIntArray]:
        if self.h is None or self.k is None or self.l is None:
            return None
        return jnp.stack((self.h, self.k, self.l), axis=1)

    def __len__(self) -> int:
        return self.length.shape[0]

    def mask_streaks(self, idxs: Indices) -> Streaks:
        """Return a new streaks container with a set of streaks discarded.

        Args:
            idxs : A set of indices of the streaks to discard.

        Returns:
            A new :class:`cbclib.Streaks` container.
        """
        return Streaks(**{attr: self[attr][idxs] for attr in self.contents()})

    def pattern_dict(self, width: float, shape: Shape, kernel: str='rectangular'
                     ) -> Union[Pattern, PatternWithHKL, PatternWithHKLID]:
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
        table = draw_line_table(lines=np.array(self.to_lines(width)), shape=shape,
                                idxs=np.array(self.idxs), kernel=kernel)
        ids, idxs = np.array(list(table)).T
        normalised_shape = (np.prod(shape[:-2], dtype=int),) + shape[-2:]
        frames, y, x = jnp.unravel_index(idxs, normalised_shape)
        vals = np.array(list(table.values()))

        if self.hkl is not None:
            h, k, l = np.array(self.hkl[ids]).T

            if self.hkl_id is not None:
                return PatternWithHKLID(ids, frames, y, x, vals, h, k, l,
                                        np.array(self.hkl_id[ids]))
            return PatternWithHKL(ids, frames, y, x, vals, h, k, l)
        return Pattern(ids, frames, y, x, vals)

    def pattern_dataframe(self, width: float, shape: Shape,
                          kernel: str='rectangular') -> pd.DataFrame:
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
        return pd.DataFrame(self.pattern_dict(width=width, shape=shape, kernel=kernel))

    def pattern_image(self, width: float, shape: Tuple[int, int],
                      kernel: str='gaussian') -> NDRealArray:
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
        return draw_line_image(np.array(self.to_lines(width)), shape=shape,
                               idxs=np.array(self.idxs), kernel=kernel)

    def pattern_mask(self, width: float, shape: Tuple[int, int], max_val: int=1,
                     kernel: str='rectangular') -> NDIntArray:
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
        return draw_line_mask(np.array(self.to_lines(width)), shape=shape, idxs=np.array(self.idxs),
                              max_val=max_val, kernel=kernel)

    def to_dataframe(self) -> pd.DataFrame:
        """Export a streaks container into :class:`pandas.DataFrame`.

        Returns:
            A dataframe with all the data specified in :class:`cbclib.Streaks`.
        """
        return pd.DataFrame({attr: self[attr] for attr in self.contents()})

    def to_lines(self, width: float) -> JaxRealArray:
        """Export a streaks container into line parameters ``x0, y0, x1, y1, width``:

        * `[x0, y0]`, `[x1, y1]` : The coordinates of the line's ends.
        * `width` : Line's width.

        Returns:
            An array of line parameters.
        """
        widths = width * jnp.ones(len(self))
        return jnp.stack((self.x0, self.y0, self.x1, self.y1, widths), axis=1)

@jax_dataclass
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
    num_threads : int = cpu_count()

    @property
    def shape(self) -> Optional[Shape]:
        if isinstance(self.transform, Crop):
            return (self.samples.size, self.transform.roi[1] - self.transform.roi[0],
                    self.transform.roi[3] - self.transform.roi[2])
        return None

    def __getitem__(self, idxs: Indices) -> CBDModel:
        return self.replace(samples=self.samples[idxs])

    def bases(self) -> JaxRealArray:
        return matmul(vecs=self.basis.mat, mats=jnp.swapaxes(self.samples.rmats, 1, 2),
                      vidxs=jnp.tile(jnp.arange(3), self.samples.size),
                      midxs=jnp.repeat(jnp.arange(self.samples.size), 3)).reshape(-1, 3, 3)

    def rec_vectors(self, hkl: JaxIntArray, hidxs: Optional[JaxIntArray]=None,
                    bidxs: Optional[JaxIntArray]=None) -> JaxRealArray:
        return matmul(jnp.asarray(hkl, dtype=float), self.bases(),
                      hidxs, bidxs, num_threads=self.num_threads)

    def filter_hkl(self, hkl: JaxIntArray) -> Tuple[JaxIntArray, JaxIntArray]:
        """Return a set of reciprocal lattice points that lie in the region of reciprocal space
        involved in diffraction.

        Args:
            hkl : Set of input Miller indices.

        Returns:
            A set of Miller indices.
        """
        hidxs = jnp.tile(jnp.arange(hkl.shape[0]), self.samples.size)
        bidxs = jnp.repeat(jnp.arange(self.samples.size), hkl.shape[0])

        rec_vec = self.rec_vectors(hkl, hidxs, bidxs)
        rec_abs = jnp.sqrt((rec_vec**2).sum(axis=-1))
        rec_th = jnp.arccos(-rec_vec[..., 2] / rec_abs)
        src_th = rec_th - jnp.arccos(0.5 * rec_abs)
        idxs = jnp.where((jnp.abs(jnp.sin(src_th)) < jnp.arccos(self.setup.kin_max[2])).T)
        return hidxs[idxs], bidxs[idxs]

    def generate_streaks(self, hkl: JaxIntArray, hidxs: Optional[JaxIntArray]=None,
                         bidxs: Optional[JaxIntArray]=None, hkl_index: bool=False) -> Streaks:
        """Generate a CBD pattern. Return a set of streaks in :class:`cbclib.Streaks` container.

        Args:
            hkl : Set of Miller indices.
            width : Width of diffraction streaks in pixels.
            hkl_index : Save ``hkl`` indices in the streaks container if True.

        Returns:
            A set of streaks, that constitute the predicted CBD pattern.
        """
        if hidxs is None:
            hidxs = cast(JaxIntArray, jnp.tile(jnp.arange(hkl.shape[0]), self.samples.size))
        if bidxs is None:
            bidxs = cast(JaxIntArray, jnp.repeat(jnp.arange(self.samples.size), hkl.shape[0]))

        rec_vec = self.rec_vectors(hkl, hidxs, bidxs)
        kin = source_lines(rec_vec, num_threads=self.num_threads,
                           kmin=self.setup.kin_min[:2], kmax=self.setup.kin_max[:2])
        mask = jnp.sum(kin, axis=(-2, -1)) > 0

        kin, rec_vec, hidxs, bidxs = kin[mask], rec_vec[mask], hidxs[mask], bidxs[mask]

        x, y = self.samples.kout_to_detector(kin + rec_vec[..., None, :], setup=self.setup,
                                             idxs=jnp.repeat(bidxs, 2),
                                             rec_vec=rec_vec[..., None, :],
                                             num_threads=self.num_threads)
        if self.transform:
            x, y = self.transform.forward_points(x, y)

        if self.shape:
            mask = (0 < y).any(axis=1) & (y < self.shape[-2]).any(axis=1) & \
                   (0 < x).any(axis=1) & (x < self.shape[-1]).any(axis=1)
            x, y, hidxs, bidxs = x[mask], y[mask], hidxs[mask], bidxs[mask]

        result = {'idxs': bidxs, 'x0': x[:, 0], 'y0': y[:, 0], 'x1': x[:, 1], 'y1': y[:, 1],
                  'h': hkl[hidxs][:, 0], 'k': hkl[hidxs][:, 1], 'l': hkl[hidxs][:, 2]}
        if hkl_index:
            result['hkl_id'] = hidxs

        return Streaks(**result)
