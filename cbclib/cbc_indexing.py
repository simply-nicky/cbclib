from __future__ import annotations
from typing import Any, Dict, Iterable, Iterator
import numpy as np
from .ini_parser import INIParser
from .data_container import DataContainer
from .bin import tilt_matrix

class ScanSetup(INIParser):
    """
    Detector tilt scan experimental setup class

    foc_pos - focus position relative to the detector [m]
    pix_size - detector pixel size [m]
    rot_axis - axis of rotation
    smp_pos - sample position relative to the detector [m]
    """
    attr_dict = {'exp_geom': ('foc_pos', 'rot_axis', 'smp_pos', 'wavelength',
                              'x_pixel_size', 'y_pixel_size', 'kin_min', 'kin_max')}
    fmt_dict = {'exp_geom': 'float'}

    foc_pos : np.ndarray
    rot_axis : np.ndarray
    smp_pos : np.ndarray
    kin_min : np.ndarray
    kin_max : np.ndarray
    wavelength : float
    x_pixel_size : float
    y_pixel_size : float

    def __init__(self, foc_pos: np.ndarray, rot_axis: np.ndarray, smp_pos: np.ndarray,
                 kin_min: np.ndarray, kin_max: np.ndarray, wavelength: float,
                 x_pixel_size: float, y_pixel_size: float) -> None:
        exp_geom={'foc_pos': foc_pos, 'rot_axis': rot_axis, 'smp_pos': smp_pos,
                  'kin_min': kin_min, 'kin_max': kin_max, 'wavelength': wavelength,
                  'x_pixel_size': x_pixel_size, 'y_pixel_size': y_pixel_size}
        super(ScanSetup, self).__init__(exp_geom=exp_geom)

    @classmethod
    def _lookup_dict(cls) -> Dict[str, str]:
        lookup = {}
        for section in cls.attr_dict:
            for option in cls.attr_dict[section]:
                lookup[option] = section
        return lookup

    def __iter__(self) -> Iterator[str]:
        return self._lookup.__ifter__()

    def __contains__(self, attr: str) -> bool:
        return attr in self._lookup

    def __repr__(self) -> str:
        return self._format(self.export_dict()).__repr__()

    def __str__(self) -> str:
        return self._format(self.export_dict()).__str__()

    def keys(self) -> Iterable[str]:
        return list(self)

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

    def _det_to_k(self, y_coords: np.ndarray, x_coords: np.ndarray, source: np.ndarray) -> np.ndarray:
        delta_y = y_coords * self.y_pixel_size - source[1]
        delta_x = x_coords * self.x_pixel_size - source[0]
        phis = np.arctan2(delta_y, delta_x)
        thetas = np.arctan(np.sqrt(delta_x**2 + delta_y**2) / source[2])
        return np.stack((np.sin(thetas) * np.cos(phis),
                         np.sin(thetas) * np.sin(phis),
                         np.cos(thetas)), axis=-1)

    def _k_to_det(self, karr: np.ndarray, source: np.ndarray) -> np.ndarray:
        theta, phi = np.arccos(karr[..., 2]), np.arctan2(karr[..., 1], karr[..., 0])
        det_x = source[2] * np.tan(theta) * np.cos(phi) + source[0]
        det_y = source[2] * np.tan(theta) * np.sin(phi) + source[1]
        return np.stack((det_y / self.y_pixel_size, det_x / self.x_pixel_size), axis=-1)

    def detector_to_kout(self, y_coords: np.ndarray, x_coords: np.ndarray) -> np.ndarray:
        return self._det_to_k(y_coords, x_coords, self.smp_pos)

    def kout_to_detector(self, kout: np.ndarray) -> np.ndarray:
        return self._k_to_det(kout, self.smp_pos)

    def detector_to_kin(self, y_coords: np.ndarray, x_coords: np.ndarray) -> np.ndarray:
        return self._det_to_k(y_coords, x_coords, self.foc_pos)

    def kin_to_detector(self, kin: np.ndarray) -> np.ndarray:
        return self._k_to_det(kin, self.foc_pos)

    def index_pts(self, streaks: np.ndarray) -> np.ndarray:
        delta_x = streaks[:, :4:2] * self.x_pixel_size - self.smp_pos[0]
        delta_y = streaks[:, 1:4:2] * self.y_pixel_size - self.smp_pos[1]
        taus_x = delta_x[:, 1] - delta_x[:, 0]
        taus_y = delta_y[:, 1] - delta_y[:, 0]
        taus_abs = taus_x**2 + taus_y**2
        products = (delta_y[:, 0] * taus_x - delta_x[:, 0] * taus_y) / taus_abs
        return np.stack((-taus_y * products, taus_x * products), axis=1)

    def tilt_matrices(self, tilts: np.ndarray) -> np.ndarray:
        return tilt_matrix(tilts, self.rot_axis)

class ScanStreaks():
    columns = {'frames', 'streaks', 'x', 'y', 'I', 'bgd'}
    init_set = {}

    def __init__(self, indices: np.ndarray) -> None:
        super(ScanStreaks, self).__init__(indices=indices)
