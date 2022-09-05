""":class:`DataContainer` class implementation.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, ItemsView, Iterator, KeysView, List, Tuple, Union, ValuesView, TypeVar
import numpy as np
from .ini_parser import INIParser
from .bin import tilt_matrix

T = TypeVar('T', bound='DataContainer')

@dataclass
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

FloatArray = Union[List[float], Tuple[float, ...], np.ndarray]

class ScanSetup(INIParser):
    """
    Detector tilt scan experimental setup class

    foc_pos - focus position relative to the detector [m]
    pix_size - detector pixel size [m]
    rot_axis - axis of rotation
    smp_pos - sample position relative to the detector [m]
    """
    attr_dict = {'exp_geom': ('foc_pos', 'rot_axis', 'wavelength', 'x_pixel_size',
                              'y_pixel_size', 'kin_min', 'kin_max')}
    fmt_dict = {'exp_geom': 'float'}

    foc_pos         : np.ndarray
    rot_axis        : np.ndarray
    kin_min         : np.ndarray
    kin_max         : np.ndarray
    wavelength      : float
    x_pixel_size    : float
    y_pixel_size    : float

    def __init__(self, foc_pos: FloatArray, rot_axis: FloatArray,
                 kin_min: FloatArray, kin_max: FloatArray, wavelength: float,
                 x_pixel_size: float, y_pixel_size: float) -> None:
        exp_geom = {'foc_pos': foc_pos, 'rot_axis': rot_axis, 'kin_min': kin_min,
                    'kin_max': kin_max, 'wavelength': wavelength,
                    'x_pixel_size': x_pixel_size, 'y_pixel_size': y_pixel_size}
        super(ScanSetup, self).__init__(exp_geom=exp_geom)

    @classmethod
    def _lookup_dict(cls) -> Dict[str, str]:
        lookup = {}
        for section in cls.attr_dict:
            for option in cls.attr_dict[section]:
                lookup[option] = section
        return lookup

    @property
    def kin_center(self) -> np.ndarray:
        return 0.5 * (self.kin_max + self.kin_min)

    def __iter__(self) -> Iterator[str]:
        return self._lookup.__iter__()

    def __contains__(self, attr: str) -> bool:
        return attr in self._lookup

    def __repr__(self) -> str:
        return self._format(self.export_dict()).__repr__()

    def __str__(self) -> str:
        return self._format(self.export_dict()).__str__()

    def keys(self) -> KeysView[str]:
        return self._lookup.keys()

    @classmethod
    def import_ini(cls, ini_file: str, **kwargs: Any) -> ScanSetup:
        """Initialize a :class:`ScanSetup` object with an
        ini file.

        Parameters
        ----------
        ini_file : str, optional
            Path to the ini file. Load the default parameters
            if None.
        **kwargs : dict
            Experimental geometry parameters.
            Initialized with `ini_file` if not provided.

        Returns
        -------
        scan_setup : ScanSetup
            A :class:`ScanSetup` object with all the attributes
            imported from the ini file.
        """
        attr_dict = cls._import_ini(ini_file)
        for option, section in cls._lookup_dict().items():
            if option in kwargs:
                attr_dict[section][option] = kwargs[option]
        return cls(**attr_dict['exp_geom'])

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

    def __repr__(self) -> str:
        return self.state_dict().__repr__()

    def __str__(self) -> str:
        return self.state_dict().__str__()

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

    def state_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

class Crop(Transform):
    """Crop transform. Crops a frame according to a region of interest.

    Attributes:
        roi : Region of interest. Comprised of four elements `[y_min, y_max,
            x_min, x_max]`.
    """
    def __init__(self, roi: Union[List[int], Tuple[int, int, int, int], np.ndarray]) -> None:
        """
        Args:
            roi : Region of interest. Comprised of four elements `[y_min, y_max,
                x_min, x_max]`.
        """
        self.roi = roi

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

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the transform as a dict.

        Returns:
            A dictionary with all the attributes.
        """
        return {'roi': self.roi[:]}

class Downscale(Transform):
    """Downscale the image by a integer ratio.

    Attributes:
        scale : Downscaling integer ratio.
    """
    def __init__(self, scale: int) -> None:
        """
        Args:
            scale : Downscaling integer ratio.
        """
        self.scale = scale

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

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the transform as a dict.

        Returns:
            A dictionary with all the attributes.
        """
        return {'scale': self.scale}

class Mirror(Transform):
    """Mirror the data around an axis.

    Attributes:
        axis : Axis of reflection.
    """
    def __init__(self, axis: int, shape: Tuple[int, int]) -> None:
        """
        Args:
            axis : Axis of reflection.
        """
        if axis not in [0, 1]:
            raise ValueError('Axis must equal to 0 or 1')
        self.axis, self.shape = axis, shape

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

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the transform as a dict.

        Returns:
            A dictionary with all the attributes.
        """
        return {'axis': self.axis, 'shape': self.shape}

class ComposeTransforms(Transform):
    """Composes several transforms together.

    Attributes:
        transforms: List of transforms.
    """
    transforms : List[Transform]

    def __init__(self, transforms: List[Transform]) -> None:
        """
        Args:
            transforms: List of transforms.
        """
        if len(transforms) < 2:
            raise ValueError('Two or more transforms are needed to compose')

        self.transforms = []
        for transform in transforms:
            pdict = transform.state_dict()
            self.transforms.append(type(transform)(**pdict))

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

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the transform as a dict.

        Returns:
            A dictionary with all the attributes.
        """
        return {'transforms': self.transforms[:]}
