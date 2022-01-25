from __future__ import annotations
from typing import Any
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
                              'x_pixel_size', 'y_pixel_size')}
    fmt_dict = {'exp_geom': 'float'}

    foc_pos : np.ndarray
    rot_axis : np.ndarray
    smp_pos : np.ndarray
    wavelength : float
    x_pixel_size : float
    y_pixel_size : float

    def __init__(self, foc_pos: np.ndarray, rot_axis: np.ndarray, smp_pos: np.ndarray,
                 wavelength: float, x_pixel_size: float, y_pixel_size: float) -> None:
        super(ScanSetup, self).__init__(exp_geom={'foc_pos': foc_pos, 'rot_axis': rot_axis,
                                                  'smp_pos': smp_pos, 'wavelength': wavelength,
                                                  'x_pixel_size': x_pixel_size,
                                                  'y_pixel_size': y_pixel_size})

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

    def detector_to_kout(self, streaks: np.ndarray) -> np.ndarray:
        delta_x = streaks[:, :4:2] * self.x_pixel_size - self.smp_pos[0]
        delta_y = streaks[..., 1:4:2] * self.y_pixel_size - self.smp_pos[1]
        phis = np.arctan2(delta_y, delta_x)
        thetas = np.arctan(np.sqrt(delta_x**2 + delta_y**2) / self.smp_pos[2])
        return np.stack((np.sin(thetas) * np.cos(phis),
                         np.sin(thetas) * np.sin(phis),
                         np.cos(thetas)), axis=-1)

    def index_pts(self, streaks: np.ndarray) -> np.ndarray:
        delta_x = streaks[:, :4:2] * self.x_pixel_size - self.smp_pos[0]
        delta_y = streaks[:, 1:4:2] * self.y_pixel_size - self.smp_pos[1]
        taus_x = delta_x[:, 1] - delta_x[:, 0]
        taus_y = delta_y[:, 1] - delta_y[:, 0]
        taus_abs = taus_x**2 + taus_y**2
        products = (delta_y[:, 0] * taus_x - delta_x[:, 0] * taus_y) / taus_abs
        return np.stack((-taus_y * products, taus_x * products), axis=1)

    def kout_to_detector(self, kout: np.ndarray) -> np.ndarray:
        theta, phi = np.arccos(kout[..., 2]), np.arctan2(kout[..., 1], kout[..., 0])
        det_x = self.smp_pos[2] * np.tan(theta) * np.cos(phi) + self.smp_pos[0]
        det_y = self.smp_pos[2] * np.tan(theta) * np.sin(phi) + self.smp_pos[1]
        return np.stack((det_x / self.x_pixel_size, det_y / self.y_pixel_size), axis=-1)

    def kin_to_detector(self, kin: np.ndarray) -> np.ndarray:
        theta, phi = np.arccos(kin[..., 2]), np.arctan2(kin[..., 1], kin[..., 0])
        det_x = self.f_pos[2] * np.tan(theta) * np.cos(phi) + self.f_pos[0]
        det_y = self.f_pos[2] * np.tan(theta) * np.sin(phi) + self.f_pos[1]
        return np.stack((det_x / self.x_pixel_size, det_y / self.y_pixel_size), axis=-1)

    def tilt_matrices(self, tilts: np.ndarray) -> np.ndarray:
        return tilt_matrix(tilts, self.rot_axis)

class ScanStreaks(DataContainer):
    attr_set = {'indices'}
    init_set = {}

    indices : np.ndarray

    def __init__(self, indices: np.ndarray) -> None:
        super(ScanStreaks, self).__init__(indices=indices)
