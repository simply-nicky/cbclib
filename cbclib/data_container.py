""":class:`DataContainer` class implementation.
"""
from __future__ import annotations
from configparser import ConfigParser
from dataclasses import dataclass
import os
import re
from typing import (Any, Callable, Dict, ItemsView, Iterator, List, Tuple, Type, Union,
                    ValuesView, TypeVar)
import numpy as np
from .bin import tilt_matrix, cartesian_to_spherical, spherical_to_cartesian

T = TypeVar('T', bound='DataContainer')

class DataContainer:
    def __getitem__(self, attr: str) -> Any:
        return self.__getattribute__(attr)

    def contents(self) -> List[str]:
        return [attr for attr in self.keys() if self.get(attr) is not None]

    def get(self, attr: str, val: Any=None) -> Any:
        if attr in self.keys():
            return self[attr]
        return val

    def keys(self) -> List[str]:
        return [attr for attr in self.__dataclass_fields__.keys()
                if getattr(type(self), attr, None) is None]

    def values(self) -> ValuesView:
        return dict(self).values()

    def items(self) -> ItemsView:
        return dict(self).items()

    def replace(self, **kwargs: Any) -> T:
        return type(self)(**dict(self, **kwargs))

I = TypeVar('I', bound='INIContainer')

class INIContainer(DataContainer):
    __ini_fields__ : Dict[str, Union[str, Tuple[str]]]

    @classmethod
    def _format_list(cls, string: str, f: Callable=str) -> List:
        is_list = re.search(r'^\[([\s\S]*)\]$', string)
        if is_list:
            return [f(p.strip('\'\"')) for p in re.split(r'\s*,\s*', is_list.group(1)) if p]
        raise ValueError(f"Invalid string: '{string}'")

    @classmethod
    def _format_tuple(cls, string: str, f: Callable=str) -> Tuple:
        is_tuple = re.search(r'^\[([\s\S]*)\]$', string)
        if is_tuple:
            return [f(p.strip('\'\"')) for p in re.split(r'\s*,\s*', is_tuple.group(1)) if p]
        raise ValueError(f"Invalid string: '{string}'")

    @classmethod
    def _format_array(cls, string: str) -> List:
        is_list = re.search(r'^\[([\s\S]*)\]$', string)
        if is_list:
            return np.fromstring(is_list.group(1), sep=',')
        raise ValueError(f"Invalid string: '{string}'")

    @classmethod
    def _format_bool(cls, string: str) -> bool:
        return string in ('yes', 'True', 'true', 'T')

    @classmethod
    def get_formatter(cls, t: str) -> Callable:
        _f1 = {'list': cls._format_list, 'List': cls._format_list,
               'tuple': cls._format_tuple, 'Tuple': cls._format_tuple}
        _f2 = {'ndarray': cls._format_array, 'float': float, 'int': int,
               'bool': cls._format_bool, 'complex': complex}
        for k1, f1 in _f1.items():
            if k1 in t:
                idx = t.index(k1) + len(k1)
                for k2, f2 in _f2.items():
                    if k2 in t[idx:]:
                        return lambda string: f1(string, f2)
                return f1
        for k2, f2 in _f2.items():
            if k2 in t:
                return f2
        return str

    @classmethod
    def _format_dict(cls, ini_dict: Dict[str, Union[str, Dict[str, str]]]) -> Dict[str, Any]:
        for attr, val in ini_dict.items():
            formatter = cls.get_formatter(cls.__dataclass_fields__[attr].type)
            if isinstance(val, dict):
                ini_dict[attr] = {k: formatter(v) for k, v in val.items()}
            if isinstance(val, str):
                ini_dict[attr] = formatter(val)
        return ini_dict

    @classmethod
    def import_ini(cls: Type[I], ini_file: str) -> I:
        if not os.path.isfile(ini_file):
            raise ValueError(f"File {ini_file} doesn't exist")
        ini_parser = ConfigParser()
        ini_parser.read(ini_file)

        ini_dict = {}
        for section, attrs in cls.__ini_fields__.items():
            if isinstance(attrs, str):
                ini_dict[attrs] = dict(ini_parser[section])
            elif isinstance(attrs, tuple):
                for attr in attrs:
                    ini_dict[attr] = ini_parser[section][attr]
            else:
                raise TypeError(f"Invalid '__ini_fields__' values: {attrs}")

        return cls(**cls._format_dict(ini_dict))

    def _get_string(self, attr) -> str:
        val = self.get(attr)
        if isinstance(val, np.ndarray):
            return np.array2string(val, separator=',')
        if isinstance(val, dict):
            return {k: str(v) for k, v in val.items()}
        return str(val)

    def ini_dict(self) -> Dict[str, Union[str, Dict[str, str]]]:
        ini_dict = {}
        for section, attrs in self.__ini_fields__.items():
            if isinstance(attrs, str):
                ini_dict[section] = self._get_string(attrs)
            if isinstance(attrs, tuple):
                ini_dict[section] = {attr: self._get_string(attr) for attr in attrs}
        return ini_dict

    def to_ini(self, ini_path: str):
        ini_parser = ConfigParser()
        for section, val in self.ini_dict().items():
            ini_parser[section] = val

        with open(ini_path, 'w') as ini_file:
            ini_parser.write(ini_file)

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

FloatArray = Union[List[float], Tuple[float, ...], np.ndarray]

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

    def tilt_matrices(self, tilts: Union[float, np.ndarray]) -> np.ndarray:
        return tilt_matrix(np.atleast_1d(tilts), self.rot_axis)

class Transform():
    """Abstract transform class."""

    def index_array(self, ss_idxs: np.ndarray, fs_idxs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def forward(self, inp: np.ndarray) -> np.ndarray:
        ss_idxs, fs_idxs = np.indices(inp.shape[-2:])
        ss_idxs, fs_idxs = self.index_array(ss_idxs, fs_idxs)
        return inp[..., ss_idxs, fs_idxs]

    def forward_points(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def backward(self, inp: np.ndarray, out: np.ndarray) -> np.ndarray:
        ss_idxs, fs_idxs = np.indices(out.shape[-2:])
        ss_idxs, fs_idxs = self.index_array(ss_idxs, fs_idxs)
        out[..., ss_idxs, fs_idxs] = inp
        return out

    def backward_points(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

@dataclass
class Crop(Transform, DataContainer):
    """Crop transform. Crops a frame according to a region of interest.

    Attributes:
        roi : Region of interest. Comprised of four elements `[y_min, y_max,
            x_min, x_max]`.
    """
    roi : Union[List[int], Tuple[int, int, int, int], np.ndarray]

    def __eq__(self, obj: object) -> bool:
        if isinstance(obj, Crop):
            return self.roi[0] == obj.roi[0] and self.roi[1] == obj.roi[1] and \
                   self.roi[2] == obj.roi[2] and self.roi[3] == obj.roi[3]
        return NotImplemented

    def __ne__(self, obj: object) -> bool:
        if isinstance(obj, Crop):
            return self.roi[0] != obj.roi[0] or self.roi[1] != obj.roi[1] or \
                   self.roi[2] != obj.roi[2] or self.roi[3] != obj.roi[3]
        return NotImplemented

    def index_array(self, ss_idxs: np.ndarray, fs_idxs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return (ss_idxs[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]],
                fs_idxs[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]])

    def forward_points(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply the transform to a set of points.

        Args:
            pts : Input array of points.

        Returns:
            Output array of points.
        """
        return x - self.roi[2], y - self.roi[0]

    def backward_points(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Tranform back an array of points.

        Args:
            pts : Input tranformed array of points.

        Returns:
            Output array of points.
        """
        return x + self.roi[2], y + self.roi[0]

@dataclass
class Downscale(Transform, DataContainer):
    """Downscale the image by a integer ratio.

    Attributes:
        scale : Downscaling integer ratio.
    """
    scale : int

    def index_array(self, ss_idxs: np.ndarray, fs_idxs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return (ss_idxs[::self.scale, ::self.scale], fs_idxs[::self.scale, ::self.scale])

    def forward_points(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply the transform to a set of points.

        Args:
            pts : Input array of points.

        Returns:
            Output array of points.
        """
        return x / self.scale, y / self.scale

    def backward_points(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Tranform back an array of points.

        Args:
            pts : Input tranformed array of points.
            shape : Detector shape.

        Returns:
            Output array of points.
        """
        return x * self.scale, y * self.scale

@dataclass
class Mirror(Transform, DataContainer):
    """Mirror the data around an axis.

    Attributes:
        axis : Axis of reflection.
    """
    axis: int
    shape: Tuple[int, int]

    def index_array(self, ss_idxs: np.ndarray, fs_idxs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.axis == 0:
            return (ss_idxs[::-1], fs_idxs[::-1])
        if self.axis == 1:
            return (ss_idxs[:, ::-1], fs_idxs[:, ::-1])
        raise ValueError('Axis must equal to 0 or 1')

    def forward_points(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply the transform to a set of points.

        Args:
            pts : Input array of points.

        Returns:
            Output array of points.
        """
        if self.axis:
            return x, self.shape[0] - y
        return self.shape[1] - x, y

    def backward_points(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Tranform back an array of points.

        Args:
            pts : Input tranformed array of points.

        Returns:
            Output array of points.
        """
        return self.forward_points(x, y)

@dataclass
class ComposeTransforms(Transform):
    """Composes several transforms together.

    Attributes:
        transforms: List of transforms.
    """
    transforms : List[Transform]

    def __post_init__(self) -> None:
        if len(self.transforms) < 2:
            raise ValueError('Two or more transforms are needed to compose')

        self.transforms = [transform.replace() for transform in self.transforms]

    def __iter__(self) -> Iterator[Transform]:
        return self.transforms.__iter__()

    def __getitem__(self, idx: Union[int, slice]) -> Union[Transform, List[Transform]]:
        return self.transforms[idx]

    def index_array(self, ss_idxs: np.ndarray, fs_idxs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        for transform in self:
            ss_idxs, fs_idxs = transform.index_array(ss_idxs, fs_idxs)
        return ss_idxs, fs_idxs

    def forward_points(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply the transform to a set of points.

        Args:
            pts : Input array of points.

        Returns:
            Output array of points.
        """
        for transform in self:
            x, y = transform.forward_points(x, y)
        return x, y

    def backward_points(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Tranform back an array of points.

        Args:
            pts : Input tranformed array of points.

        Returns:
            Output array of points.
        """
        for transform in list(self)[::-1]:
            x, y = transform.backward_points(x, y)
        return x, y
