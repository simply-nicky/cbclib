from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy.ndimage import label, center_of_mass, mean
from .bin import (FFTW, empty_aligned, median_filter, gaussian_filter, gaussian_grid,
                  gaussian_grid_grad, cartesian_to_spherical, spherical_to_cartesian,
                  euler_matrix, calc_source_lines)
from .data_processing import ScanSetup, Crop, Streaks, Indices

IntArray = Union[List[int], Tuple[int, ...], np.ndarray]

class ScanStreaks():
    columns = {'frames', 'streaks', 'x', 'y', 'length', 'width', 'p', 'I_raw', 'sgn', 'bgd'}

    def __init__(self, dataframe: pd.DataFrame, setup: ScanSetup) -> None:
        if not self.columns.issubset(dataframe.columns):
            raise ValueError(f'Dataframe must contain the following columns: {self.columns}')

        self._df, self._setup = dataframe, setup

    @classmethod
    def import_csv(cls, path: str, setup: ScanSetup) -> ScanStreaks:
        return cls(pd.read_csv(path, usecols=cls.columns), setup)

    @classmethod
    def import_hdf(cls, path: str, key: str, setup: ScanSetup) -> ScanStreaks:
        return cls(pd.read_hdf(path, key, usecols=cls.columns), setup)

    @property
    def dtype(self) -> np.dtype:
        return self._df.dtypes['sgn']

    def __repr__(self) -> str:
        return self._df.__repr__()

    def _repr_html_(self) -> Optional[str]:
        return self._df._repr_html_()

    def __str__(self) -> str:
        return self._df.__str__()

    def get_dataframe(self) -> pd.DataFrame:
        return self._df

    def get_crop(self) -> Crop:
        return Crop((self._df['y'].min(), self._df['y'].max(), self._df['x'].min(), self._df['x'].max()))

    def create_qmap(self, thetas: Dict[int, float], qx_arr: np.ndarray, qy_arr: np.ndarray,
                    qz_arr: np.ndarray) -> Map3D:
        q_map = np.zeros((qx_arr.size, qy_arr.size, qz_arr.size), dtype=self.dtype)

        for frame, df_frame in self._df.groupby('frames'):
            q_frame = self._setup.detector_to_kout(df_frame.loc[:, 'y'],
                                                   df_frame.loc[:, 'x']) - self._setup.kin_center
            q_frame = q_frame.dot(self._setup.tilt_matrices(thetas[frame])[0].T)
            x_idxs = np.searchsorted(qx_arr, q_frame[:, 0])
            y_idxs = np.searchsorted(qy_arr, q_frame[:, 1])
            z_idxs = np.searchsorted(qz_arr, q_frame[:, 2])
            mask = (x_idxs < qx_arr.size) & (y_idxs < qy_arr.size) & (z_idxs < qz_arr.size)
            q_map[x_idxs[mask], y_idxs[mask], z_idxs[mask]] += df_frame.loc[:, 'p'][mask]
        return Map3D(q_map, qx_arr, qy_arr, qz_arr)

    def pattern_dataframe(self, frame: int, crop: Optional[Crop]=None) -> pd.DataFrame:
        if crop is None:
            crop = self.get_crop()

        df_frame = self._df[self._df['frames'] == frame]
        df_frame['x'], df_frame['y'] = crop.forward_points(df_frame['x'], df_frame['y'])
        mask = (df_frame['y'] < crop.roi[1] - crop.roi[0]) & (df_frame['x'] < crop.roi[3] - crop.roi[2])
        return df_frame[mask]

    def pattern_image(self, frame: int, crop: Optional[Crop]=None) -> np.ndarray:
        df_frame = self.pattern_dataframe(frame, crop)
        pattern = np.zeros((crop.roi[1] - crop.roi[0], crop.roi[3] - crop.roi[2]), dtype=self.dtype)
        pattern[df_frame['y'], df_frame['x']] = df_frame['p']
        return pattern

    def pattern_sparse(self, frame: int, crop: Optional[Crop]=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        df_frame = self.pattern_dataframe(frame, crop)
        return tuple(df_frame[['x', 'y', 'p']].to_numpy().T)

@dataclass
class Map3D():
    val : np.ndarray
    x   : np.ndarray
    y   : np.ndarray
    z   : np.ndarray

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.val.shape

    @staticmethod
    def _find_reduced(vectors: np.ndarray, basis: np.ndarray) -> np.ndarray:
        """Find reduced vector to basis set.

        Args:
            vectors : array of vectors of shape (N, 3).
            basis : basis set of shape (M, 3).
        """
        prod = vectors.dot(basis.T)
        mask = 2 * np.abs(prod) < (basis * basis).sum(axis=1)
        return np.where(mask.all(axis=1))[0]

    def is_compatible(self, map_3d: Map3D) -> bool:
        return ((self.x == map_3d.x).all() and (self.y == map_3d.y).all() and
                (self.z == map_3d.z).all())

    def __getitem__(self, idxs: Tuple[Indices, Indices, Indices]) -> Map3D:
        if np.asarray(idxs[0]).ndim > 1 or np.asarray(idxs[1]).ndim > 1 or np.asarray(idxs[2]).ndim > 1:
            raise ValueError('Indices must be 0- or 1-dimensional')
        idxs = dict(zip(range(3), idxs))
        for axis, size in zip(idxs, self.shape):
            if isinstance(idxs[axis], (int, slice)):
                idxs[axis] = np.atleast_1d(np.arange(size)[idxs[axis]])
        return Map3D(val=self.val[np.meshgrid(*idxs.values(), indexing='ij')],
                     x=self.x[idxs[0]], y=self.y[idxs[1]], z=self.z[idxs[2]])

    def __add__(self, obj: Union[Map3D, float, int]) -> Map3D:
        if isinstance(obj, (float, int)):
            return Map3D(self.val + obj, self.x, self.y, self.z)
        if isinstance(obj, Map3D) and self.is_compatible(obj):
            return Map3D(self.val + obj.val, self.x, self.y, self.z)
        raise ValueError("Can't sum two incompatible Map3D objects")

    def __sub__(self, obj: Union[Map3D, float, int]) -> Map3D:
        if isinstance(obj, (float, int)):
            return Map3D(self.val - obj, self.x, self.y, self.z)
        if isinstance(obj, Map3D) and self.is_compatible(obj):
            return Map3D(self.val - obj.val, self.x, self.y, self.z)
        raise ValueError("Can't subtract two incompatible Map3D objects")

    def __prod__(self, obj: Union[Map3D, float, int]) -> Map3D:
        if isinstance(obj, (float, int)):
            return Map3D(self.val / obj, self.x, self.y, self.z)
        if isinstance(obj, Map3D) and self.is_compatible(obj):
            return Map3D(self.val * obj.val, self.x, self.y, self.z)
        raise ValueError("Can't multiply two incompatible Map3D objects")

    def __truediv__(self, obj: Union[Map3D, float, int]) -> Map3D:
        if isinstance(obj, (float, int)):
            return Map3D(self.val / obj, self.x, self.y, self.z)
        if isinstance(obj, Map3D) and self.is_compatible(obj):
            return Map3D(self.val / obj.val, self.x, self.y, self.z)
        raise ValueError("Can't divide two incompatible Map3D objects")

    def clip(self, vmin: float, vmax: float) -> Map3D:
        return Map3D(np.clip(self.val, vmin, vmax), self.x, self.y, self.z)

    def criterion(self, x: np.ndarray, Sig: float, sig: float, epsilon: float=1e-12,
                  num_threads: int=1) -> Tuple[float, np.ndarray]:
        y_hat, hkl = gaussian_grid(self.x, self.y, self.z, x[:9].reshape((3, 3)),
                                   Sig, sig, num_threads)
        grad = gaussian_grid_grad(self.x, self.y, self.z, x[:9].reshape((3, 3)),
                                  hkl, Sig, sig, num_threads)
        return (-np.sum(self.val * np.log(y_hat + epsilon)),
                -np.sum(self.val / (y_hat + epsilon) * grad, axis=(1, 2, 3)))

    def gaussian_blur(self, sigma: Union[float, Tuple[float, float, float]],
                      num_threads: int=1) -> Map3D:
        val = gaussian_filter(self.val, sigma, num_threads=num_threads)
        return Map3D(val, self.x, self.y, self.z)

    def fft(self, num_threads: int=1) -> Map3D:
        val = empty_aligned(self.shape, dtype='complex64')
        fft_obj = FFTW(self.val.astype(np.complex64), val, threads=num_threads,
                       axes=(0, 1, 2), flags=('FFTW_ESTIMATE',))

        kx = np.fft.fftshift(np.fft.fftfreq(self.shape[0], d=self.x[1] - self.x[0]))
        ky = np.fft.fftshift(np.fft.fftfreq(self.shape[1], d=self.y[1] - self.y[0]))
        kz = np.fft.fftshift(np.fft.fftfreq(self.shape[2], d=self.z[1] - self.z[0]))
        val = np.fft.fftshift(np.abs(fft_obj())) / np.sqrt(np.prod(self.shape))
        return Map3D(val, kx, ky, kz)

    def find_peaks(self, val: float, reduce: bool=False) -> np.ndarray:
        mask = self.val > val
        peak_lbls, peak_num = label(mask)
        index = np.arange(1, peak_num + 1)
        peaks = np.array(center_of_mass(self.val, labels=peak_lbls, index=index))
        means = np.array(mean(self.val, labels=peak_lbls, index=index))
        idxs = np.argsort(means)[::-1]
        peaks = peaks[idxs]

        if reduce:
            reduced = peaks[[1]] - peaks[0]
            idxs = self._find_reduced(peaks - peaks[0], reduced)
            reduced = np.concatenate((reduced, peaks[idxs[[1]]] - peaks[0]))
            idxs = self._find_reduced(peaks - peaks[0], reduced)
            reduced = np.concatenate((reduced, peaks[idxs[[1]]] - peaks[0]))
            return reduced + peaks[0]

        return peaks

    def white_tophat(self, structure: np.ndarray, num_threads: int=1) -> Map3D:
        val = median_filter(self.val, footprint=structure, num_threads=num_threads)
        return Map3D(self.val - val, self.x, self.y, self.z)

@dataclass
class Basis():
    mat : np.ndarray

    def __post_init__(self):
        self.mat = self.mat.reshape((3, 3))

    @classmethod
    def import_spherical(cls, sph_mat: np.ndarray) -> Basis:
        return cls(spherical_to_cartesian(sph_mat))

    def to_spherical(self) -> np.ndarray:
        return cartesian_to_spherical(self.mat)

    def rotate(self, rot_mat: np.ndarray) -> Basis:
        return Basis(self.mat.dot(rot_mat.T))

    def rotate_euler(self, angles: np.ndarray) -> Basis:
        return self.rotate(euler_matrix(angles))

    def tilt(self, tilt: float, setup: ScanSetup) -> Basis:
        return self.rotate(setup.tilt_matrices(tilt))

    def reciprocate(self, scan_setup: ScanSetup) -> Basis:
        a_rec = np.cross(self.mat[1], self.mat[2]) / (np.cross(self.mat[1], self.mat[2]).dot(self.mat[0]))
        b_rec = np.cross(self.mat[2], self.mat[0]) / (np.cross(self.mat[2], self.mat[0]).dot(self.mat[1]))
        c_rec = np.cross(self.mat[0], self.mat[1]) / (np.cross(self.mat[0], self.mat[1]).dot(self.mat[2]))
        return Basis(np.stack((a_rec, b_rec, c_rec)) * scan_setup.wavelength)

@dataclass
class CBDModel():
    basis   : Basis
    setup   : ScanSetup

    def generate_hkl(self, q_abs: float) -> np.ndarray:
        lat_size = np.rint(q_abs / self.basis.to_spherical()[:, 0]).astype(int)
        h_idxs = np.arange(-lat_size[0], lat_size[0] + 1)
        k_idxs = np.arange(-lat_size[1], lat_size[1] + 1)
        l_idxs = np.arange(-lat_size[2], lat_size[2] + 1)
        h_grid, k_grid, l_grid = np.meshgrid(h_idxs, k_idxs, l_idxs)
        hkl = np.stack((h_grid.ravel(), k_grid.ravel(), l_grid.ravel()), axis=1)

        rec_vec = hkl.dot(self.basis.mat)
        rec_abs = np.sqrt((rec_vec**2).sum(axis=-1))
        rec_th = np.arccos(-rec_vec[..., 2] / rec_abs)

        mask = np.abs(np.sin(rec_th - np.arccos(0.5 * rec_abs))) < np.sqrt(self.setup.kin_max[0]**2 +
                                                                           self.setup.kin_max[1]**2)
        mask &= (rec_abs != 0.0) & (rec_abs < q_abs)
        return hkl[mask]

    def generate_streaks(self, hkl: np.ndarray, width: float, crop: Optional[Crop]=None,
                         num_threads: int=1) -> Streaks:
        kin, hkl = calc_source_lines(basis=self.basis.mat, hkl=hkl, kin_min=self.setup.kin_min,
                                     kin_max=self.setup.kin_max, threads=num_threads)
        kout = kin + hkl.dot(self.basis.mat)[:, None]

        x, y = self.setup.kout_to_detector(kout)
        if crop:
            x, y = crop.forward_points(x, y)

        streaks = np.stack((x[:, 0], y[:, 0], x[:, 1], y[:, 1],
                            width * np.ones(hkl.shape[0])), axis=1)
        return Streaks(streaks)

@dataclass
class IndexProblem:
    tol        : Tuple[float, float]
    sph_mat    : np.ndarray
    hkl        : np.ndarray
    width      : float
    scan_setup : ScanSetup
    crop       : Crop
    ij         : np.ndarray
    p          : np.ndarray
    idxs       : np.ndarray = field(init=False)
    epsilon    : float = field(default=1e-12)
    ratio      : float = field(default=1.0 / 255.0, repr=False)

    def __post_init__(self):
        self.idxs = np.arange(self.ij.size)

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.crop.roi[1] - self.crop.roi[0], self.crop.roi[3] - self.crop.roi[2])

    def generate_basis(self, x: np.ndarray) -> Basis:
        new_mat = np.concatenate(((self.sph_mat[:, 0] * x[:3])[:, None], self.sph_mat[:, 1:]), axis=1)
        return Basis.import_spherical(new_mat).rotate_euler(x[3:6])

    def pattern_image(self, x: np.ndarray) -> np.ndarray:
        model = CBDModel(self.generate_basis(x), self.scan_setup)
        fstreaks = model.generate_streaks(self.hkl, self.width, self.crop)
        pattern = fstreaks.pattern_image(self.shape, max_val=255, profile='linear')
        return pattern * self.ratio

    def pattern_sparse(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        model = CBDModel(self.generate_basis(x), self.scan_setup)
        fstreaks = model.generate_streaks(self.hkl, self.width, self.crop)
        x, y, q = fstreaks.pattern_sparse(self.shape, max_val=255, profile='linear')
        ij = (x + self.shape[0] * y).astype(int)
        return ij, q * self.ratio

    def fitness(self, x: np.ndarray) -> List[float]:
        ij, q = self.pattern_sparse(x)
        _, p_int, q_int = np.intersect1d(self.ij, ij, return_indices=True)
        p_diff = np.setdiff1d(self.idxs, p_int, assume_unique=True)
        return [-np.sum(self.p[p_int] * np.log(q[q_int] + self.epsilon))
                -np.log(self.epsilon) * np.sum(self.p[p_diff]),]

    def get_bounds(self) -> tuple:
        return ([1.0 - self.tol[0],] * 3 + [-self.tol[1],] * 3,
                [1.0 + self.tol[0],] * 3 + [self.tol[1],] * 3)
