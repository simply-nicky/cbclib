from ini_parser import INIParser
import numpy as np

class ScanSetup(INIParser):
    """
    Detector tilt scan experimental setup class

    foc_pos - focus position relative to the detector [m]
    pix_size - detector pixel size [m]
    rot_axis - axis of rotation
    smp_pos - sample position relative to the detector [m]
    """
    attr_dict = {'exp_geom': ('foc_pos', 'pix_size', 'rot_axis', 'smp_pos')}
    fmt_dict = {'exp_geom': 'float'}

    def __init__(self, foc_pos, pix_size, rot_axis, smp_pos):
        super(ScanSetup, self).__init__(foc_pos=foc_pos, pix_size=pix_size,
                                        rot_axis=rot_axis, smp_pos=smp_pos)

    def detector_to_kout(self, streaks):
        delta_x = streaks[..., 0] * self.pix_size - self.smp_pos[0]
        delta_y = streaks[..., 1] * self.pix_size - self.smp_pos[1]
        phis = np.arctan2(delta_y, delta_x)
        thetas = np.arctan(np.sqrt(delta_x**2 + delta_y**2) / self.smp_pos[2])
        return np.stack((np.sin(thetas) * np.cos(phis),
                         np.sin(thetas) * np.sin(phis),
                         np.cos(thetas)), axis=-1)

    def kout_to_detector(self, kout):
        theta, phi = np.arccos(kout[..., 2]), np.arctan2(kout[..., 1], kout[..., 0])
        det_x = self.smp_pos[2] * np.tan(theta) * np.cos(phi)
        det_y = self.smp_pos[2] * np.tan(theta) * np.sin(phi)
        return (np.stack((det_x, det_y), axis=-1) + self.smp_pos[:2]) / self.pix_size

    def kin_to_detector(self, kin):
        theta, phi = np.arccos(kin[..., 2]), np.arctan2(kin[..., 1], kin[..., 0])
        det_x = self.f_pos[2] * np.tan(theta) * np.cos(phi)
        det_y = self.f_pos[2] * np.tan(theta) * np.sin(phi)
        return (np.stack((det_x, det_y), axis=-1) + self.f_pos[:2]) / self.pix_size
