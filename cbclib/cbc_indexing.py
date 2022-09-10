from __future__ import annotations
from typing import (Any, ClassVar, Dict, ItemsView, Iterable, Iterator, KeysView, List,
                    Optional, Set, Tuple, Union, ValuesView)
from dataclasses import InitVar, dataclass, field
from weakref import ReferenceType, ref
import numpy as np
import pandas as pd
from scipy.ndimage import label, center_of_mass, mean
from .bin import (FFTW, empty_aligned, median_filter, gaussian_filter, gaussian_grid,
                  gaussian_grid_grad,  euler_angles, euler_matrix, tilt_angles,
                  tilt_matrix, cross_entropy, calc_source_lines)
from .data_container import Crop, DataContainer, ScanSetup, Basis
from .data_processing import Streaks
from .cxi_protocol import Indices

IntArray = Union[List[int], Tuple[int, ...], np.ndarray]

@dataclass
class ScanStreaks():
    columns         : ClassVar[Set[str]] = {'frames', 'x', 'y', 'p', 'I_raw', 'sgn', 'bgd'}
    hkl_columns     : ClassVar[Set[str]] = {'h', 'k', 'l'}
    dataframe       : pd.DataFrame
    setup           : ScanSetup
    crop            : Optional[Crop] = None

    def __post_init__(self):
        if not self.columns.issubset(self.dataframe.columns):
            raise ValueError(f'Dataframe must contain the following columns: {self.columns}')
        for col in self.hkl_columns:
            if col not in self.dataframe.columns:
                self.dataframe[col] = np.nan
        if self.crop is None:
            self.crop = self.get_crop()
        if not self.isunique():
            self.reset_index()

    @classmethod
    def import_csv(cls, path: str, setup: ScanSetup) -> ScanStreaks:
        return cls(pd.read_csv(path, usecols=cls.columns), setup)

    @classmethod
    def import_hdf(cls, path: str, key: str, setup: ScanSetup) -> ScanStreaks:
        return cls(pd.read_hdf(path, key, usecols=cls.columns), setup)

    @property
    def dtype(self) -> np.dtype:
        return self.dataframe.dtypes['sgn']

    def _repr_html_(self) -> Optional[str]:
        return self.dataframe._repr_html_()

    def create_qmap(self, samples: ScanSamples, qx_arr: np.ndarray, qy_arr: np.ndarray,
                    qz_arr: np.ndarray) -> Map3D:
        q_map = np.zeros((qx_arr.size, qy_arr.size, qz_arr.size), dtype=self.dtype)
        first_frame = self.dataframe['frames'].min()

        for frame, df_frame in self.dataframe.groupby('frames'):
            kout = samples[frame].detector_to_kout(df_frame['x'], df_frame['y'], self.setup)
            q_frame = kout - self.setup.kin_center
            q_frame = q_frame.dot(samples.rotation(frame, first_frame)[0])
            x_idxs = np.searchsorted(qx_arr, q_frame[:, 0])
            y_idxs = np.searchsorted(qy_arr, q_frame[:, 1])
            z_idxs = np.searchsorted(qz_arr, q_frame[:, 2])
            mask = (x_idxs < qx_arr.size) & (y_idxs < qy_arr.size) & (z_idxs < qz_arr.size)
            np.add.at(q_map, (x_idxs[mask], y_idxs[mask], z_idxs[mask]), df_frame['p'][mask])
        return Map3D(q_map, qx_arr, qy_arr, qz_arr)

    def drop_duplicates(self) -> pd.DataFrame:
        return ScanStreaks(self.dataframe.drop_duplicates(['frames', 'x', 'y']), self.setup)

    def get_crop(self) -> Crop:
        return Crop((self.dataframe['y'].min(), self.dataframe['y'].max(),
                     self.dataframe['x'].min(), self.dataframe['x'].max()))

    def isunique(self) -> bool:
        return np.unique(self.dataframe.index).size == self.dataframe.index.size

    def pattern_dataframe(self, frame: int) -> pd.DataFrame:
        df_frame = self.dataframe[self.dataframe['frames'] == frame]
        mask = (self.crop.roi[0] < df_frame['y']) & (df_frame['y'] < self.crop.roi[1]) & \
               (self.crop.roi[2] < df_frame['x']) & (df_frame['x'] < self.crop.roi[3])
        df_frame = df_frame[mask]
        df_frame['x'], df_frame['y'] = self.crop.forward_points(df_frame['x'], df_frame['y'])
        return df_frame

    def pattern_dict(self, frame: int) -> Dict[str, np.ndarray]:
        return {col: np.asarray(val) for col, val in self.pattern_dataframe(frame).to_dict(orient='list').items()}

    def pattern_image(self, frame: int) -> np.ndarray:
        df_frame = self.pattern_dataframe(frame)
        pattern = np.zeros((self.crop.roi[1] - self.crop.roi[0],
                            self.crop.roi[3] - self.crop.roi[2]), dtype=self.dtype)
        pattern[df_frame['y'], df_frame['x']] = df_frame['p']
        return pattern

    def refine_indexing(self, frame: int, tol: Tuple[float, float, float], basis: Basis, sample: Sample,
                        q_abs: float, width: float) -> IndexProblem:
        return IndexProblem(parent=ref(self), tol=tol, basis=basis, sample=sample, q_abs=q_abs,
                            width=width, pattern=self.pattern_dict(frame))

    def reset_index(self):
        self.dataframe.reset_index(drop=True, inplace=True)

@dataclass
class Map3D():
    val : np.ndarray
    x   : np.ndarray
    y   : np.ndarray
    z   : np.ndarray

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.val.shape

    @property
    def coordinate(self) -> np.ndarray:
        return np.stack((self.x, self.y, self.z))

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

    def __getitem__(self, idxs: Tuple[Indices, Indices, Indices]) -> Map3D:
        if np.asarray(idxs[0]).ndim > 1 or np.asarray(idxs[1]).ndim > 1 or np.asarray(idxs[2]).ndim > 1:
            raise ValueError('Indices must be 0- or 1-dimensional')
        idxs = dict(zip(range(3), idxs))
        for axis, size in zip(idxs, self.shape):
            if isinstance(idxs[axis], (int, slice)):
                idxs[axis] = np.atleast_1d(np.arange(size)[idxs[axis]])
        return Map3D(val=self.val[np.meshgrid(*idxs.values(), indexing='ij')],
                     x=self.x[idxs[0]], y=self.y[idxs[1]], z=self.z[idxs[2]])

    def __add__(self, obj: Any) -> Map3D:
        if np.isscalar(obj):
            return Map3D(self.val + obj, self.x, self.y, self.z)
        if isinstance(obj, Map3D):
            if not self.is_compatible(obj):
                raise TypeError("Can't sum two incompatible Map3D objects")
            return Map3D(self.val + obj.val, self.x, self.y, self.z)
        return NotImplemented

    def __sub__(self, obj: Any) -> Map3D:
        if np.isscalar(obj):
            return Map3D(self.val - obj, self.x, self.y, self.z)
        if isinstance(obj, Map3D):
            if not self.is_compatible(obj):
                raise TypeError("Can't subtract two incompatible Map3D objects")
            return Map3D(self.val - obj.val, self.x, self.y, self.z)
        return NotImplemented

    def __rmul__(self, obj: Any) -> Map3D:
        if np.isscalar(obj):
            return Map3D(obj * self.val, self.x, self.y, self.z)
        return NotImplemented

    def __mul__(self, obj: Any) -> Map3D:
        if np.isscalar(obj):
            return Map3D(obj * self.val, self.x, self.y, self.z)
        if isinstance(obj, Map3D):
            if not self.is_compatible(obj):
                raise TypeError("Can't multiply two incompatible Map3D objects")
            return Map3D(self.val * obj.val, self.x, self.y, self.z)
        return NotImplemented

    def __truediv__(self, obj: Any) -> Map3D:
        if np.isscalar(obj):
            return Map3D(self.val / obj, self.x, self.y, self.z)
        if isinstance(obj, Map3D):
            if self.is_compatible(obj):
                raise TypeError("Can't divide two incompatible Map3D objects")
            return Map3D(self.val / obj.val, self.x, self.y, self.z)
        return NotImplemented

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
            return self.index_to_coord(reduced + peaks[0])

        return self.index_to_coord(peaks)

    def index_to_coord(self, index: np.ndarray) -> np.ndarray:
        idx = np.rint(index).astype(int)
        crd = np.take(self.coordinate, idx)
        return crd + (index - idx) * (np.take(self.coordinate, idx + 1) - crd)

    def is_compatible(self, map_3d: Map3D) -> bool:
        return ((self.x == map_3d.x).all() and (self.y == map_3d.y).all() and
                (self.z == map_3d.z).all())

    def white_tophat(self, structure: np.ndarray, num_threads: int=1) -> Map3D:
        val = median_filter(self.val, footprint=structure, num_threads=num_threads)
        return Map3D(self.val - val, self.x, self.y, self.z)

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

@dataclass
class CBDModel():
    basis   : Basis
    sample  : Sample
    crop    : Crop
    setup   : ScanSetup

    def __post_init__(self):
        self.basis = self.sample.rotate(self.basis)

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.crop.roi[1] - self.crop.roi[0], self.crop.roi[3] - self.crop.roi[2])

    def generate_hkl(self, q_abs: float) -> np.ndarray:
        lat_size = np.rint(q_abs / self.basis.to_spherical()[:, 0]).astype(int)
        h_idxs = np.arange(-lat_size[0], lat_size[0] + 1)
        k_idxs = np.arange(-lat_size[1], lat_size[1] + 1)
        l_idxs = np.arange(-lat_size[2], lat_size[2] + 1)
        h_grid, k_grid, l_grid = np.meshgrid(h_idxs, k_idxs, l_idxs)
        hkl = np.stack((h_grid.ravel(), k_grid.ravel(), l_grid.ravel()), axis=1)
        hkl = np.compress(hkl.any(axis=1), hkl, axis=0)

        rec_vec = hkl.dot(self.basis.mat)
        rec_abs = np.sqrt((rec_vec**2).sum(axis=-1))
        rec_th = np.arccos(-rec_vec[..., 2] / rec_abs)

        mask = np.abs(np.sin(rec_th - np.arccos(0.5 * rec_abs))) < np.sqrt(self.setup.kin_max[0]**2 +
                                                                           self.setup.kin_max[1]**2)
        mask &= (rec_abs < q_abs)
        return hkl[mask]

    def generate_streaks(self, hkl: np.ndarray, width: float) -> Streaks:
        kin, mask = calc_source_lines(basis=self.basis.mat, hkl=hkl, kin_min=self.setup.kin_min,
                                      kin_max=self.setup.kin_max)
        idxs = np.arange(hkl.shape[0])[mask]
        kout = kin + hkl[idxs].dot(self.basis.mat)[:, None]

        x, y = self.sample.kout_to_detector(kout, self.setup)
        mask = (self.crop.roi[0] < y).any(axis=1) & (y < self.crop.roi[1]).any(axis=1) & \
               (self.crop.roi[2] < x).any(axis=1) & (x < self.crop.roi[3]).any(axis=1)
        (x, y), idxs = self.crop.forward_points(x[mask], y[mask]), idxs[mask]
        return Streaks(x0=x[:, 0], y0=y[:, 0], x1=x[:, 1], y1=y[:, 1], width=width * np.ones(x.shape[0]),
                       h=hkl[idxs, 0], k=hkl[idxs, 1], l=hkl[idxs, 2], hkl_index=idxs)

    def pattern_dataframe(self, hkl: np.ndarray, width: float, dp: float=1e-3,
                          profile: str='linear') -> pd.DataFrame:
        streaks = self.generate_streaks(hkl, width)
        return streaks.pattern_dataframe(self.shape, dp=dp, profile=profile)

@dataclass
class IndexProblem:
    parent    : ReferenceType[ScanStreaks]
    tol       : InitVar[Tuple[float, float]]
    basis     : Basis
    sample    : Sample
    q_abs     : InitVar[float]
    width     : float
    pattern   : InitVar[Dict[str, np.ndarray]]

    crop      : Crop = field(init=False)
    hkl       : np.ndarray = field(init=False)
    setup     : ScanSetup = field(init=False)

    q_max     : ClassVar[int] = 1000
    epsilon   : ClassVar[float] = 1e-12
    footprint : ClassVar[np.ndarray] = np.array([[False, False,  True, False, False],
                                                 [False,  True,  True,  True, False],
                                                 [ True,  True,  True,  True,  True],
                                                 [False,  True,  True,  True, False],
                                                 [False, False,  True, False, False]], dtype=bool)

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.crop.roi[1] - self.crop.roi[0], self.crop.roi[3] - self.crop.roi[2])

    def __getstate__(self) -> Dict[str, Any]:
        state = {key: val for key, val in self.__dict__.items() if key != 'parent'}
        state['parent'] = None
        return state

    def __post_init__(self, tol: Tuple[float, float, float], q_abs: float, pattern: Dict[str, np.ndarray]):
        self.setup, self.crop = self.parent().setup, self.parent().crop
        self._ij, self._p = self._pattern_to_ij_and_q(pattern)
        self._idxs = np.arange(self._ij.size)

        _lbounds, _rbounds = [], []
        self._slices = [None, None, None]

        if tol[0]:
            _lbounds += [-tol[0],] * 3
            _rbounds += [tol[0],] * 3
            self._slices[0] = slice(len(_lbounds) - 3, len(_lbounds))
        if tol[1]:
            _lbounds += [1.0 - tol[0],] * 3
            _rbounds += [1.0 + tol[0],] * 3
            self._slices[1] = slice(len(_lbounds) - 3, len(_lbounds))
        if tol[2]:
            _lbounds += [1.0 - tol[0],] * 2
            _rbounds += [1.0 + tol[0],] * 2
            self._slices[2] = slice(len(_lbounds) - 2, len(_lbounds))

        self._bounds = (_lbounds, _rbounds)
        self.x0 = np.mean(self._bounds, axis=0)
        self.update_hkl(self.x0, q_abs)

    def _get_slice(self, x: np.ndarray, i: int) -> np.ndarray:
        if self._slices[i]:
            return x[self._slices[i]]
        return (np.zeros(3), np.ones(3), np.ones(2))[i]

    def _pattern_to_ij_and_q(self, pattern: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        return pattern['x'] + self.shape[1] * pattern['y'], pattern['p']

    def generate_sample(self, x: np.ndarray) -> Sample:
        return Sample(Rotation.import_euler(*self._get_slice(x, 0)) * self.sample.rotation,
                      self.sample.pos * self._get_slice(x, 1))

    def generate_setup(self, x: np.ndarray) -> ScanSetup:
        foc_pos = np.copy(self.setup.foc_pos)
        foc_pos[:2] *= self._get_slice(x, 2)
        return self.setup.replace(foc_pos=foc_pos)

    def generate_model(self, x: np.ndarray) -> CBDModel:
        return CBDModel(self.basis, self.generate_sample(x), self.crop, self.generate_setup(x))

    def generate_streaks(self, x: np.ndarray) -> Streaks:
        return self.generate_model(x).generate_streaks(self.hkl, self.width)

    def pattern_dataframe(self, x: np.ndarray) -> pd.DataFrame:
        return self.generate_model(x).pattern_dataframe(hkl=self.hkl, width=self.width,
                                                        dp=1.0 / self.q_max, profile='linear')

    def pattern_image(self, x: np.ndarray) -> np.ndarray:
        streaks = self.generate_streaks(x)
        return streaks.pattern_image(self.shape, dp=1.0 / self.q_max, profile='linear')

    def pattern_mask(self, x: np.ndarray) -> np.ndarray:
        streaks = self.generate_streaks(x)
        return streaks.pattern_mask(self.shape, max_val=self.q_max, profile='linear')

    def pattern_dict(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        streaks = self.generate_streaks(x)
        return streaks.pattern_dict(self.shape, dp=1.0 / self.q_max, profile='linear')

    def fitness(self, x: np.ndarray) -> List[float]:
        return [cross_entropy(x=self._ij, p=self._p, q=self.pattern_mask(x).ravel(),
                              q_max=self.q_max, epsilon=self.epsilon),]

    def get_bounds(self) -> Tuple[List[float], List[float]]:
        return self._bounds

    def update_hkl(self, x: np.ndarray, q_abs: float) -> None:
        self.hkl = self.generate_model(x).generate_hkl(q_abs)

    def index_frame(self, x: np.ndarray, frame: int, iterations: int=4, num_threads: int=1):
        if self.parent().crop != self.crop:
            raise ValueError('Parent Crop object has been changed, '\
                             'please create new IndexProblem instance.')
        if self.parent().setup != self.setup:
            raise ValueError('Parent ScanSetup object has been changed, '\
                             'please create new IndexProblem instance.')

        df_frame = self.parent().pattern_dataframe(frame)
        df_frame['ij'] = self._pattern_to_ij_and_q(df_frame)[0]
        df_frame = df_frame.sort_values('ij')

        df = self.pattern_dataframe(x)
        mask = np.zeros(self.shape, dtype=bool)
        labels = np.zeros(self.shape, dtype=np.uint32)
        mask[df_frame['y'], df_frame['x']] = True
        labels[df['y'], df['x']] = df['hkl_index']
        for _ in range(iterations):
            labels = median_filter(labels, footprint=self.footprint, inp_mask=labels,
                                   mask=mask, num_threads=num_threads)

        ij_sim = np.where(labels.ravel())[0]
        df_idxs = df_frame.index[df_frame['ij'].searchsorted(ij_sim)]
        self.parent().dataframe.loc[df_idxs, ['h', 'k', 'l']] = self.hkl[labels.ravel()[ij_sim], :]

MapSamples = Union[Iterable[Tuple[int, Sample]], Dict[int, Sample]]

class ScanSamples:
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

    def __str__(self) -> str:
        return self._dct.__str__()

    def __repr__(self) -> str:
        return self._dct.__repr__()

    def keys(self) -> KeysView[str]:
        return self._dct.keys()

    def values(self) -> ValuesView[Sample]:
        return self._dct.values()

    def items(self) -> ItemsView[str, Sample]:
        return self._dct.items()

    def to_dict(self) -> Dict[str, Sample]:
        return dict(self._dct)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame((sample.to_dataframe() for sample in self.values()), index=self.keys())

    def rotation(self, from_frames: Union[int, Iterable[int]], to_frames: Union[int, Iterable[int]]) -> List[Rotation]:
        from_frames, to_frames = np.atleast_1d(from_frames), np.atleast_1d(to_frames)
        rotations = []
        for (f1, f2) in zip(from_frames, to_frames):
            rotations.append(self[f2].rotation * self[f1].rotation.reciprocate())
        return rotations

    def get_positions(self, axis: int) -> np.ndarray:
        return np.asarray([sample.pos[axis] for sample in self.values()])

    def set_positions(self, positions: np.ndarray, axis: int) -> ScanSamples:
        obj = ScanSamples(self.items())
        for frame in obj:
            obj[frame].pos[axis] = positions[frame]
        return obj
