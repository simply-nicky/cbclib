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

    def __init__(self, foc_pos, rot_axis, smp_pos, wavelength, x_pixel_size,
                 y_pixel_size):
        super(ScanSetup, self).__init__(exp_geom={'foc_pos': foc_pos, 'rot_axis': rot_axis,
                                                  'smp_pos': smp_pos, 'wavelength': wavelength,
                                                  'x_pixel_size': x_pixel_size,
                                                  'y_pixel_size': y_pixel_size})

    @classmethod
    def import_ini(cls, ini_file, **kwargs):
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

    def detector_to_kout(self, streaks):
        delta_x = streaks[:, :4:2] * self.x_pixel_size - self.smp_pos[0]
        delta_y = streaks[..., 1:4:2] * self.y_pixel_size - self.smp_pos[1]
        phis = np.arctan2(delta_y, delta_x)
        thetas = np.arctan(np.sqrt(delta_x**2 + delta_y**2) / self.smp_pos[2])
        return np.stack((np.sin(thetas) * np.cos(phis),
                         np.sin(thetas) * np.sin(phis),
                         np.cos(thetas)), axis=-1)

    def index_pts(self, streaks):
        delta_x = streaks[:, :4:2] * self.x_pixel_size - self.smp_pos[0]
        delta_y = streaks[:, 1:4:2] * self.y_pixel_size - self.smp_pos[1]
        taus_x = delta_x[:, 1] - delta_x[:, 0]
        taus_y = delta_y[:, 1] - delta_y[:, 0]
        taus_abs = taus_x**2 + taus_y**2
        products = (delta_y[:, 0] * taus_x - delta_x[:, 0] * taus_y) / taus_abs
        return np.stack((-taus_y * products, taus_x * products), axis=1)

    def kout_to_detector(self, kout):
        theta, phi = np.arccos(kout[..., 2]), np.arctan2(kout[..., 1], kout[..., 0])
        det_x = self.smp_pos[2] * np.tan(theta) * np.cos(phi) + self.smp_pos[0]
        det_y = self.smp_pos[2] * np.tan(theta) * np.sin(phi) + self.smp_pos[1]
        return np.stack((det_x / self.x_pixel_size, det_y / self.y_pixel_size), axis=-1)

    def kin_to_detector(self, kin):
        theta, phi = np.arccos(kin[..., 2]), np.arctan2(kin[..., 1], kin[..., 0])
        det_x = self.f_pos[2] * np.tan(theta) * np.cos(phi) + self.f_pos[0]
        det_y = self.f_pos[2] * np.tan(theta) * np.sin(phi) + self.f_pos[1]
        return np.stack((det_x / self.x_pixel_size, det_y / self.y_pixel_size), axis=-1)

    def tilt_matrices(self, tilts):
        return tilt_matrix(tilts, self.rot_axis)

class ScanStreaks(DataContainer):
    attr_set = {'streaks', 'tilts'}
    init_set = {'kin', 'kout', 'tilt_mats', 'scat_vec'}

    def __init__(self, scan_setup, **kwargs):
        self.__dict__['scan_setup'] = scan_setup

        super(ScanStreaks, self).__init__(**kwargs)

        self._init_dict()

    def _init_dict(self):
        if self.tilt_mats is None:
            self.tilt_mats = self.scan_setup.tilt_matrices(-self.tilts)
        if self.kin is None:
            self.kin = {idx: np.tile([[[0., 0., 1.]]],
                                     (streaks.shape[0], 2, 1)).dot(self.tilt_mats[idx])
                        for idx, streaks in self.streaks.items()}
        if self.kout is None:
            self.kout = {idx: self.scan_setup.detector_to_kout(streaks).dot(self.tilt_mats[idx])
                         for idx, streaks in self.streaks.items()}
        if self.scat_vec is None:
            self.scat_vec = {idx: self.kout[idx] - self.kin[idx] for idx in self.streaks}
