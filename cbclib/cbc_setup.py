from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, ItemsView, Iterable, Iterator, KeysView, List, Tuple, Union, ValuesView
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel
from .bin import (tilt_matrix, cartesian_to_spherical, spherical_to_cartesian, euler_angles,
                  euler_matrix, tilt_angles)
from .data_container import DataContainer, INIContainer

@dataclass
class Basis(INIContainer):
    __ini_fields__ = {'basis': ('a_vec', 'b_vec', 'c_vec')}

    a_vec : np.ndarray
    b_vec : np.ndarray
    c_vec : np.ndarray

    def __post_init__(self):
        self.mat = np.stack((self.a_vec, self.b_vec, self.c_vec))

    @classmethod
    def import_matrix(cls, mat: np.ndarray) -> Basis:
        return cls(mat[0], mat[1], mat[2])

    @classmethod
    def import_spherical(cls, sph_mat: np.ndarray) -> Basis:
        return cls.import_matrix(spherical_to_cartesian(sph_mat))

    def reciprocate(self, scan_setup: ScanSetup) -> Basis:
        a_rec = np.cross(self.b_vec, self.c_vec) / (np.cross(self.b_vec, self.c_vec).dot(self.a_vec))
        b_rec = np.cross(self.c_vec, self.a_vec) / (np.cross(self.c_vec, self.a_vec).dot(self.b_vec))
        c_rec = np.cross(self.a_vec, self.b_vec) / (np.cross(self.a_vec, self.b_vec).dot(self.c_vec))
        return Basis(*(np.stack((a_rec, b_rec, c_rec)) * scan_setup.wavelength))

    def to_spherical(self) -> np.ndarray:
        return cartesian_to_spherical(self.mat)

@dataclass
class ScanSetup(INIContainer):
    """
    Detector tilt scan experimental setup class

    foc_pos - focus position relative to the detector [m]
    pix_size - detector pixel size [m]
    rot_axis - axis of rotation
    smp_pos - sample position relative to the detector [m]
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
        delta_y = y * self.y_pixel_size - source[1]
        delta_x = x * self.x_pixel_size - source[0]
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

    def detector_to_kout(self, x: np.ndarray, y: np.ndarray, smp_pos: np.ndarray) -> np.ndarray:
        return self._det_to_k(x, y, smp_pos)

    def kout_to_detector(self, kout: np.ndarray, smp_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self._k_to_det(kout, smp_pos)

    def detector_to_kin(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self._det_to_k(x, y, self.foc_pos)

    def kin_to_detector(self, kin: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self._k_to_det(kin, self.foc_pos)

    def tilt_rotation(self, theta: float) -> Rotation:
        return Rotation.import_tilt(theta, np.arccos(self.rot_axis[2]), np.arctan2(self.rot_axis[1], self.rot_axis[0]))

@dataclass
class Rotation(DataContainer):
    matrix : np.ndarray = np.eye(3, 3)

    def __post_init__(self):
        self.matrix = self.matrix.reshape((3, 3))

    @classmethod
    def import_euler(cls, alpha: float, beta: float, gamma: float) -> Rotation:
        return Rotation(euler_matrix(np.array([alpha, beta, gamma])))

    @classmethod
    def import_tilt(cls, theta: float, alpha: float, beta: float) -> Rotation:
        return Rotation(tilt_matrix(np.array([theta, alpha, beta])))

    def __call__(self, inp: np.ndarray) -> np.ndarray:
        return inp.dot(self.matrix.T)

    def __mul__(self, obj: Any) -> Rotation:
        if isinstance(obj, Rotation):
            return Rotation(self.matrix.dot(obj.matrix))
        return NotImplemented

    def reciprocate(self) -> Rotation:
        return Rotation(self.matrix.T)

    def to_euler(self) -> np.ndarray:
        return euler_angles(self.matrix)

    def to_tilt(self) -> np.ndarray:
        if np.allclose(self.matrix, self.matrix.T):
            eigw, eigv = np.linalg.eigh(self.matrix)
            axis = eigv[np.isclose(eigw, 1.0)]
            theta = np.arccos(0.5 * (np.trace(self.matrix) - 1.0))
            return np.array([theta, np.arccos(axis[0, 2]), np.arctan2(axis[0, 1], axis[0, 0])])
        return tilt_angles(self.matrix)

@dataclass
class Sample():
    rotation : Rotation
    pos : np.ndarray
    mat_columns : ClassVar[Tuple[str]] = ('Rxx', 'Rxy', 'Rxz',
                                          'Ryx', 'Ryy', 'Ryz',
                                          'Rzx', 'Rzy', 'Rzz')
    pos_columns : ClassVar[Tuple[str]] = ('x' , 'y' , 'z' )

    @classmethod
    def import_dataframe(cls, dataframe: pd.Series) -> Sample:
        return cls(rotation=Rotation(dataframe[list(cls.mat_columns)].to_numpy()),
                   pos=dataframe[list(cls.pos_columns)].to_numpy())

    def rotate(self, basis: Basis) -> Basis:
        return Basis.import_matrix(self.rotation(basis.mat))

    def detector_to_kout(self, x: np.ndarray, y: np.ndarray, setup: ScanSetup) -> np.ndarray:
        return setup.detector_to_kout(x, y, self.pos)

    def kout_to_detector(self, kout: np.ndarray, setup: ScanSetup) -> Tuple[np.ndarray, np.ndarray]:
        return setup.kout_to_detector(kout, self.pos)

    def to_dataframe(self) -> pd.Series:
        return pd.Series(np.concatenate((self.rotation.matrix.ravel(), self.pos)),
                         index=self.mat_columns + self.pos_columns)

MapSamples = Union[Iterable[Tuple[int, Sample]], Dict[int, Sample]]

class ScanSamples():
    def __init__(self, items: MapSamples=list()):
        self._dct = dict(items)

    @classmethod
    def import_dataframe(cls, df: pd.DataFrame) -> ScanSamples:
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
        return self._dct.keys()

    def values(self) -> ValuesView[Sample]:
        return self._dct.values()

    def items(self) -> ItemsView[str, Sample]:
        return self._dct.items()

    def get_positions(self, axis: int) -> np.ndarray:
        return np.asarray([sample.pos[axis] for sample in self.values()])

    def regularise(self, kernel_bandwidth: Tuple[int, int]) -> ScanSamples:
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
        from_frames, to_frames = np.atleast_1d(from_frames), np.atleast_1d(to_frames)
        rotations = []
        for (f1, f2) in zip(from_frames, to_frames):
            rotations.append(self[f2].rotation * self[f1].rotation.reciprocate())
        return rotations

    def set_positions(self, positions: np.ndarray, axis: int) -> ScanSamples:
        obj = deepcopy(self)
        for frame in obj:
            obj[frame].pos[axis] = positions[frame]
        return obj

    def to_dict(self) -> Dict[str, Sample]:
        return dict(self._dct)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame((sample.to_dataframe() for sample in self.values()), index=self.keys())
