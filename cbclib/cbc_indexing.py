from __future__ import annotations
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple, Union
from dataclasses import InitVar, dataclass, field
from weakref import ref
import numpy as np
import pandas as pd
from scipy.ndimage import label, center_of_mass, mean
from tqdm.auto import tqdm

from .bin import (FFTW, empty_aligned, median_filter, gaussian_filter, gaussian_grid,
                  gaussian_grid_grad, cross_entropy, calc_source_lines, filter_hkl,
                  unique_indices, find_kins, kr_grid, update_sf, scaling_criterion,
                  xtal_interpolate)
from .cbc_setup import Basis, Rotation, Sample, ScanSamples, ScanSetup, Streaks
from .cxi_protocol import Indices
from .data_container import DataContainer, Transform, Crop, ReferenceType

@dataclass
class CBCTable():
    """Convergent beam crystallography tabular data. The data is stored in :class:`pandas.DataFrame`
    table. A table must contain the following columns:

    * `frames` : Frame index.
    * `index` : Diffraction streak index.
    * `x`, `y` : x and y pixel coordinate.
    * `p` : Normalised pattern value. The value lies in (0.0 - 1.0) interval.
    * `I_raw` : Raw photon count.
    * `sgn` : Background subtracted intensity.
    * `bgd` : Background intensity.
    * `h`, `k`, `l` : Miller indices.

    Args:
        table : CBC tabular data.
        setup : Experimental setup.
        crop : Detector region of interest.
    """
    columns         : ClassVar[Set[str]] = {'frames', 'x', 'y', 'p', 'I_raw', 'sgn', 'bgd'}
    table           : pd.DataFrame
    setup           : ScanSetup
    crop            : Optional[Crop] = None

    def __post_init__(self):
        if not self.columns.issubset(self.table.columns):
            raise ValueError(f'Dataframe must contain the following columns: {self.columns}')
        if self.crop is None:
            self.crop = self.get_crop()

    @classmethod
    def import_csv(cls, path: str, setup: ScanSetup) -> CBCTable:
        """Initialize a CBC table with a CSV file ``path`` and an experimental geometry object
        ``setup``.

        Args:
            path : Path to the CSV file.
            setup : Experimental geometry.

        Returns:
            A new CBC table object.
        """
        return cls(pd.read_csv(path, usecols=cls.columns), setup)

    @classmethod
    def import_hdf(cls, path: str, key: str, setup: ScanSetup) -> CBCTable:
        """Initialize a CBC table with data saved in a HDF5 file ``path`` at a ``key`` key inside
        the file and an experimental geometry object ``setup``.

        Args:
            path : Path to the CSV file.
            setup : Experimental geometry.

        Returns:
            A new CBC table object.
        """
        return cls(pd.read_hdf(path, key, usecols=cls.columns), setup)

    @property
    def dtype(self) -> np.dtype:
        return self.table.dtypes['sgn']

    def _repr_html_(self) -> Optional[str]:
        return self.table._repr_html_()

    def create_qmap(self, samples: ScanSamples, qx_arr: np.ndarray, qy_arr: np.ndarray,
                    qz_arr: np.ndarray) -> Map3D:
        """Map the measured normalised intensities to the reciprocal space. Returns a
        :class:`cbclib.Map3D` 3D data container capable of performing the auto Fourier indexing.

        Args:
            samples : Set of scan samples.
            qx_arr : Array of reciprocal x coordinates.
            qy_arr : Array of reciprocal y coordinates.
            qz_arr : Array of reciprocal z coordinates.

        Returns:
            3D data container of measured normalised intensities.
        """
        q_map = np.zeros((qx_arr.size, qy_arr.size, qz_arr.size), dtype=self.dtype)

        for frame, df_frame in self.table.groupby('frames'):
            kout = samples[frame].detector_to_kout(df_frame['x'], df_frame['y'], self.setup)
            q_frame = kout - self.setup.kin_center
            q_frame = samples[frame].rotation(q_frame)
            x_idxs = np.searchsorted(qx_arr, q_frame[:, 0])
            y_idxs = np.searchsorted(qy_arr, q_frame[:, 1])
            z_idxs = np.searchsorted(qz_arr, q_frame[:, 2])
            mask = (x_idxs < qx_arr.size) & (y_idxs < qy_arr.size) & (z_idxs < qz_arr.size)
            np.add.at(q_map, (x_idxs[mask], y_idxs[mask], z_idxs[mask]), df_frame['p'][mask])
        return Map3D(q_map, qx_arr, qy_arr, qz_arr)

    def drop_duplicates(self) -> pd.DataFrame:
        """Discard the pixel data, that has duplicate `x`, `y` coordinates.

        Returns:
            New CBC table with the duplicate data discarded.
        """
        return CBCTable(self.table.drop_duplicates(['frames', 'x', 'y']), self.setup)

    def get_crop(self) -> Crop:
        """Return the region of interest on the detector plane.

        Returns:
            A new crop object with the ROI, inferred from the CBC table.
        """
        return Crop((self.table['y'].min(), self.table['y'].max() + 1,
                     self.table['x'].min(), self.table['x'].max() + 1))

    def generate_kins(self, basis: Basis, samples: ScanSamples, num_threads: int=1) -> np.ndarray:
        """Convert diffraction pattern locations to incoming wavevectors. The incoming wavevectors
        are normalised and specify the spatial frequencies of the incoming beam that bring about
        the diffraction signal measured on the detector.

        Args:
            basis : Basis vectors of crystal lattice unit cell.
            samples : Set of scan samples.
            num_threads : Number of threads used in the calculations.

        Raises:
            AttributeError : If Miller indices ('h', 'k', 'l') are not present in the CBC table.

        Returns:
            A set of incoming wavevectors.
        """
        if not {'h', 'k', 'l'}.issubset(self.table.columns):
            raise AttributeError('CBC table is not indexed.')
        frames, fidxs, _ = unique_indices(frames=self.table['frames'].to_numpy(),
                                          indices=self.table['index'].to_numpy())
        smp_df = samples.to_dataframe()
        return find_kins(x=self.table['x'].to_numpy(), y=self.table['y'].to_numpy(),
                         hkl=self.table[['h', 'k', 'l']].to_numpy(), fidxs=fidxs,
                         smp_pos=smp_df.iloc[frames, 9:].to_numpy(),
                         rot_mat=smp_df.iloc[frames, :9].to_numpy(), basis=basis.mat,
                         x_pixel_size=self.setup.x_pixel_size,
                         y_pixel_size=self.setup.y_pixel_size, num_threads=num_threads)

    def pattern_dataframe(self, frame: int) -> pd.DataFrame:
        """Return a single pattern table. The `x`, `y` coordinates are transformed by the ``crop``
        attribute.

        Args:
            frame : Frame index.

        Returns:
            A :class:`pandas.DataFrame` table.
        """
        df_frame = self.table[self.table['frames'] == frame]
        mask = (self.crop.roi[0] < df_frame['y']) & (df_frame['y'] < self.crop.roi[1]) & \
               (self.crop.roi[2] < df_frame['x']) & (df_frame['x'] < self.crop.roi[3])
        df_frame = df_frame[mask]
        df_frame['x'], df_frame['y'] = self.crop.forward_points(df_frame['x'], df_frame['y'])
        return df_frame

    def pattern_dict(self, frame: int) -> Dict[str, np.ndarray]:
        """Return a single pattern table in :class:`dict` format. The `x`, `y` coordinates are
        transformed by the ``crop`` attribute.

        Args:
            frame : Frame index.

        Returns:
            A pattern table in :class:`dict` format.
        """
        dataframe = self.pattern_dataframe(frame).to_dict(orient='list')
        return {col: np.asarray(val) for col, val in dataframe.items()}

    def pattern_image(self, frame: int) -> np.ndarray:
        """Return a CBC pattern image array. The `x`, `y` coordinates are transformed by the
        ``crop`` attribute.

        Args:
            frame : Frame index.

        Returns:
            A pattern image array.
        """
        df_frame = self.pattern_dataframe(frame)
        pattern = np.zeros((self.crop.roi[1] - self.crop.roi[0],
                            self.crop.roi[3] - self.crop.roi[2]), dtype=self.dtype)
        pattern[df_frame['y'], df_frame['x']] = df_frame['p']
        return pattern

    def refine(self, frame: int, bounds: Tuple[float, float, float], basis: Basis,
               sample: Sample, q_abs: float, width: float, alpha: float=0.0) -> SampleProblem:
        """Return a :class:`SampleProblem` problem designed to perform the sample refinement.
        Indexing refinement yields a :class:`Sample` object, that attains the best fit between the
        predicted pattern and the pattern from this CBC table.

        Args:
            frame : Frame index.
            bounds : Sample refinement bounds. A tuple of three bounds (`ang_bound`, `pos_bound`,
                `foc_bound`), where each elements is as following:

                * `ang_bound` : Sample orientation bound. Sample orientation is given by a rotation
                  matrix :class:`cbclib.Rotation` ``rotation``. To perform the orientation
                  refinement, ``rotation`` is multiplied by another rotation matrix given by three
                  Euler angles, that lie in interval :code:`[-ang_bound, ang_bound]`.
                * `pos_bound` : Sample position bound. Sample position is refined in the interval
                  of coordinates :code:`[pos * (1.0 - pos_bound), pos * (1.0 + pos_bound)]`, where
                  `pos` is the initial sample position.
                * `foc_bound` : Focal point bound. Focal point is refined in the interval of
                  coordinates :code:`[foc_pos * (1.0 - foc_bound), foc_pos * (1.0 + foc_bound)]`,
                  where `foc_pos` is the initial focal point.

                Each of the terms is diregarded in the refinement process, if the corresponding
                bound is equal to 0.
            basis : Basis vectors of crystal lattice unit cell.
            sample : Sample object. The object is given by the rotation matrix and a sample
                position.
            q_abs : Size of the recpirocal space. Reciprocal vectors are normalised, and span in
                [0.0 - 1.0] interval.
            width : Diffraction streak width in pixels. The value is used to generate a predicted
                CBC pattern.
            alpha : Regularisation term in the loss function.

        Returns:
            A CBC sample refinement problem.

        See Also:
            cbclib.SampleProblem : CBC sample refinement problem.
        """
        return SampleProblem(bounds=bounds, basis=basis, sample=sample, q_abs=q_abs, width=width,
                             parent=ref(self), pattern=self.pattern_dict(frame), alpha=alpha)

    def scale(self, xtal_shape: Tuple[int, int], basis: Basis, samples: ScanSamples,
              num_threads: int=1) -> IntensityScaler:
        """Return a :class:`IntensityScaler` CBC dataset intensity scaler. The scaler generates a
        crystal diffraction efficiency map and structure factors based on diffraction signal. The
        intensity scaling algorithm uses the crystal basis and sample objects to project the
        diffraction signal onto the crystal plane.

        Args:
            xtal_shape : Crystal plane grid shape.
            basis : Basis vectors of crystal lattice unit cell.
            samples : Set of scan samples.
            num_threads : Number of threads to be used in the calculations.

        Raises:
            AttributeError : If Miller indices ('h', 'k', 'l') are not present in the CBC table.

        Returns:
            A CBC dataset intensity scaler.
        """
        if not {'h', 'k', 'l'}.issubset(self.table.columns):
            raise AttributeError('CBC table is not indexed.')
        frames, fidxs, iidxs = unique_indices(frames=self.table['frames'].to_numpy(),
                                              indices=self.table['index'].to_numpy())
        hkl = self.table.iloc[iidxs[:-1]][['h', 'k', 'l']]
        hkl, hkl_idxs = np.unique(hkl, return_inverse=True, axis=0)
        kins = self.generate_kins(basis, samples, num_threads)
        xtal_map = (np.asarray(xtal_shape) - 1) * (kins - self.setup.kin_min[:2]) / \
                   (self.setup.kin_max[:2] - self.setup.kin_min[:2])
        return IntensityScaler(parent=ref(self), frames=frames, fidxs=fidxs, iidxs=iidxs,
                               hkl=hkl, hkl_idxs=hkl_idxs, xtal_map=xtal_map,
                               signal=self.table['sgn'].to_numpy(), num_threads=num_threads)

    def update_crop(self, crop: Optional[Crop]=None) -> CBCTable:
        """Return a new CBC table with the updated region of interest.

        Args:
            crop : A new region of interest.

        Returns:
            A new CBC table with the updated ROI.
        """
        return CBCTable(self.table, self.setup, crop)

@dataclass
class Map3D():
    """3D data object designed to perform Fourier auto-indexing. Projects measured intensities to
    the reciprocal space and provides several tools to works with a 3D data in the reciprocal
    space. The container uses the `FFTW <https://www.fftw.org>`_ C library to perform the
    3-dimensional Fourier transform.

    Args:
        val : 3D data array.
        x : x coordinates.
        y : y coordinates.
        z : z coordinates.
    """
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
        """Clip the 3D data in a range of values :code:`[vmin, vmax]`.

        Args:
            vmin : Lower bound.
            vmax : Upper bound.

        Returns:
            A new 3D data object.
        """
        return Map3D(np.clip(self.val, vmin, vmax), self.x, self.y, self.z)

    def criterion(self, x: np.ndarray, envelope: float, sigma: float, epsilon: float=1e-12,
                  num_threads: int=1) -> Tuple[float, np.ndarray]:
        """Return a marginal log-likelihood, that the 3D data is represented by a given set of
        basis vectors.

        Args:
            x : Flattened matrix of basis vectors.
            envelope : A width of the envelope gaussian function.
            sigma : A width of diffraction orders.
            epsilon : Epsilon value in the log term.
            num_threads : Number of threads used in the computations.
        """
        y_hat, hkl = gaussian_grid(x_arr=self.x, y_arr=self.y, z_arr=self.z,
                                   basis=x[:9].reshape((3, 3)), envelope=envelope,
                                   sigma=sigma, threads=num_threads)
        grad = gaussian_grid_grad(x_arr=self.x, y_arr=self.y, z_arr=self.z,
                                  basis=x[:9].reshape((3, 3)), hkl=hkl, envelope=envelope,
                                  sigma=sigma, threads=num_threads)
        return (-np.sum(self.val * np.log(y_hat + epsilon)),
                -np.sum(self.val / (y_hat + epsilon) * grad, axis=(1, 2, 3)))

    def gaussian_blur(self, sigma: Union[float, Tuple[float, float, float]],
                      num_threads: int=1) -> Map3D:
        """Apply Gaussian blur to the 3D data.

        Args:
            sigma : width of the gaussian blur.
            num_threads : Number of threads used in the calculations.

        Returns:
            A new 3D data object with blurred out data.
        """
        val = gaussian_filter(self.val, sigma, num_threads=num_threads)
        return Map3D(val, self.x, self.y, self.z)

    def fft(self, num_threads: int=1) -> Map3D:
        """Perform 3D Fourier transform. `FFTW <https://www.fftw.org>`_ C library is used to
        compute the transform.

        Args:
            num_threads : Number of threads used in the calculations.

        Returns:
            A 3Ddata object with the Fourier image data.

        See Also:
            cbclib.bin.FFTW : Python wrapper of FFTW library.
        """
        val = empty_aligned(self.shape, dtype='complex64')
        fft_obj = FFTW(self.val.astype(np.complex64), val, threads=num_threads,
                       axes=(0, 1, 2), flags=('FFTW_ESTIMATE',))

        kx = np.fft.fftshift(np.fft.fftfreq(self.shape[0], d=self.x[1] - self.x[0]))
        ky = np.fft.fftshift(np.fft.fftfreq(self.shape[1], d=self.y[1] - self.y[0]))
        kz = np.fft.fftshift(np.fft.fftfreq(self.shape[2], d=self.z[1] - self.z[0]))
        val = np.fft.fftshift(np.abs(fft_obj())) / np.sqrt(np.prod(self.shape))
        return Map3D(val, kx, ky, kz)

    def find_peaks(self, val: float, reduce: bool=False) -> np.ndarray:
        """Find a set of basis vectors, that correspond to the peaks in the 3D data, that lie
        above the threshold ``val``.

        Args:
            val : Threshold value.
            reduce : Reduce a set of peaks to three basis vectors if True.

        Returns:
            A set of peaks above the threshold if ``reduce`` is False, a set of three basis
            vectors otherwise.
        """
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
        """Transform a set of data indices to a set of coordinates.

        Args:
            index : An array of indices.

        Returns:
            An array of coordinates.
        """
        idx = np.rint(index).astype(int)
        crd = np.take(self.coordinate, idx)
        return crd + (index - idx) * (np.take(self.coordinate, idx + 1) - crd)

    def is_compatible(self, map_3d: Map3D) -> bool:
        """Check if 3D data object has a compatible set of coordinates.

        Args:
            map_3d : 3D data object.

        Returns:
            True if the 3D data object ``map_3d`` has a compatible set of coordinates.
        """
        return ((self.x == map_3d.x).all() and (self.y == map_3d.y).all() and
                (self.z == map_3d.z).all())

    def white_tophat(self, structure: np.ndarray, num_threads: int=1) -> Map3D:
        """Perform 3-dimensional white tophat filtering.

        Args:
            structure : Structuring element used for the filter.
            num_threads : Number of threads used in the calculations.

        Returns:
            New 3D data container with the filtered data.
        """
        val = median_filter(self.val, footprint=structure, num_threads=num_threads)
        return Map3D(self.val - val, self.x, self.y, self.z)

@dataclass
class CBDModel():
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
    sample      : Sample
    setup       : ScanSetup
    transform   : Optional[Transform] = None
    shape       : Optional[Tuple[int, int]] = None

    def __post_init__(self):
        self.basis = self.sample.rotate(self.basis)
        if isinstance(self.transform, Crop):
            self.shape = (self.transform.roi[1] - self.transform.roi[0],
                          self.transform.roi[3] - self.transform.roi[2])

    def generate_hkl(self, q_abs: float) -> np.ndarray:
        """Return a set of reciprocal lattice points inside the sphere of radius ``q_abs``,
        that can fall into the Bragg condition.

        Args:
            q_abs : Sphere radius. Reciprocal vectors are normalised and lie in [0.0 - 1.0]
                interval.

        Returns:
            A set of Miller indices.
        """
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
        src_th = np.abs(np.sin(rec_th - np.arccos(0.5 * rec_abs)))

        mask = src_th < np.sqrt(self.setup.kin_max[0]**2 + self.setup.kin_max[1]**2)
        mask &= (rec_abs < q_abs)
        return hkl[mask]

    def generate_streaks(self, hkl: np.ndarray, width: float) -> Streaks:
        """Generate a CBD pattern. Return a set of streaks in :class:`cbclib.Streaks` container.

        Args:
            hkl : Set of Miller indices.
            width : Width of diffraction streaks in pixels.

        Returns:
            A set of streaks, that constitute the predicted CBD pattern.
        """
        kin, mask = calc_source_lines(basis=self.basis.mat, hkl=hkl, kin_min=self.setup.kin_min,
                                      kin_max=self.setup.kin_max)
        hkl = hkl[mask]
        kout = kin + hkl.dot(self.basis.mat)[:, None]

        x, y = self.sample.kout_to_detector(kout, self.setup)
        if self.transform:
            x, y = self.transform.forward_points(x, y)

        if self.shape:
            mask = (0 < y).any(axis=1) & (y < self.shape[0]).any(axis=1) & \
                   (0 < x).any(axis=1) & (x < self.shape[1]).any(axis=1)
            x, y, hkl = x[mask], y[mask], hkl[mask]
        return Streaks(x0=x[:, 0], y0=y[:, 0], x1=x[:, 1], y1=y[:, 1],
                       h=hkl[:, 0], k=hkl[:, 1], l=hkl[:, 2],
                       width=width * np.ones(x.shape[0]))

    def filter_streaks(self, hkl: np.ndarray, signal: np.ndarray, background: np.ndarray,
                       width: float,  threshold: float=0.95, dp: float=1e-3, profile: str='tophat',
                       num_threads: int=1) -> Streaks:
        """Generate a predicted pattern and filter out all the streaks, which signal-to-noise ratio
        is below the ``threshold``.

        Args:
            hkl : Set of reciprocal lattice point to use for prediction.
            signal : Measured signal.
            background : Measured background, standard deviation of the signal is the square root of
                background.
            width : Difrraction streak width in pixels of a predicted pattern.
            threshold : SNR threshold.
            dp : The quantisation step of a predicted pattern.
            profile : Line width profiles of generated streaks. The following keyword values are
                allowed:

                * `tophat` : Top-hat (rectangular) function profile.
                * `linear` : Linear (triangular) function profile.
                * `quad` : Quadratic (parabola) function profile.

            num_threads : Number of threads used in the calculations.

        Returns:
            A set of filtered out streaks, SNR of which is above the ``threshold``.
        """
        streaks = self.generate_streaks(hkl, width)
        pattern = streaks.pattern_dataframe(self.shape, dp=dp, profile=profile)
        mask = filter_hkl(sgn=signal, bgd=background, coord=pattern[['x', 'y']].to_numpy(),
                          prob=pattern['p'].to_numpy(), idxs=pattern['index'].to_numpy(),
                          threshold=threshold, num_threads=num_threads)
        return streaks.mask_streaks(mask)

    def pattern_dataframe(self, hkl: np.ndarray, width: float, dp: float=1e-3,
                          profile: str='linear') -> pd.DataFrame:
        """Predict a CBD pattern and return in the :class:`pandas.DataFrame` format.

        Args:
            hkl : Set of reciprocal lattice point to use for prediction.
            width : Difrraction streak width in pixels of a predicted pattern.
            dp : Likelihood value increment.
            profile : Line width profiles. The following keyword values are allowed:

                * `tophat` : Top-hat (rectangular) function profile.
                * `linear` : Linear (triangular) function profile.
                * `quad` : Quadratic (parabola) function profile.

        Returns:
            A pattern in :class:`pandas.DataFrame` format.
        """
        streaks = self.generate_streaks(hkl, width)
        return streaks.pattern_dataframe(self.shape, dp=dp, profile=profile)

@dataclass
class SampleProblem():
    """Sample refinement problem. It employs :class:`cbclib.CBDModel` CBD pattern prediction
    to find sample position and alignment, that yields the best fit with the experimentally
    measured pattern. The criterion calculates the marginal log-likelihood, that the
    experimentally measured pattern corresponds to the predicted one.

    Args:
        parent : Reference to the parent CBC table.
        bounds : Sample refinement bounds. A tuple of three bounds (`ang_bound`, `pos_bound`,
            `foc_bound`), where each elements is as following:

            * `ang_bound` : Sample orientation bound. Sample orientation is given by a rotation
              matrix :class:`cbclib.Rotation` ``rotation``. To perform the orientation
              refinement, ``rotation`` is multiplied by another rotation matrix given by three
              Euler angles, that lie in interval :code:`[-ang_bound, ang_bound]`.
            * `pos_bound` : Sample position bound. Sample position is refined in the interval
              of coordinates :code:`[pos * (1.0 - pos_bound), pos * (1.0 + pos_bound)]`, where
              `pos` is the initial sample position.
            * `foc_bound` : Focal point bound. Focal point is refined in the interval of
              coordinates :code:`[foc_pos * (1.0 - foc_bound), foc_pos * (1.0 + foc_bound)]`,
              where `foc_pos` is the initial focal point.

            Each of the terms is diregarded in the refinement process, if the corresponding
            bound is equal to 0.
        basis : Basis vectors of lattice unit cell.
        sample : Sample object. The object is given by the rotation matrix and a sample
            position.
        q_abs : Size of the recpirocal space. Reciprocal vectors are normalised, and span in
            [0.0 - 1.0] interval.
        width : Diffraction streak width in pixels. The value is used to generate a predicted
            CBC pattern.
        pattern : Experimentally measured CBD pattern.
        alpha : Regularisation term in the loss function.

    Attributes:
        alpha : Regularisation term in the loss function.
        basis : Basis vectors of lattice unit cell.
        crop : Detector region of interest.
        hkl : Set of reciprocal lattice points used in the prediction.
        parent : Reference to the parent CBC table.
        sample : Sample object. The object is given by the rotation matrix and a sample
            position.
        setup : Experimental geometry.
        width : Diffraction streak width in pixels. The value is used to generate a predicted
            CBC pattern.
        x0 : Initial solution.
    """
    parent    : ReferenceType[CBCTable]
    bounds    : InitVar[Tuple[float, float, float]]
    basis     : Basis
    sample    : Sample
    q_abs     : InitVar[float]
    width     : float
    pattern   : InitVar[Dict[str, np.ndarray]]
    alpha     : float = field(default=0.0)

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

    def __post_init__(self, bounds: Tuple[float, float, float], q_abs: float,
                      pattern: Dict[str, np.ndarray]):
        self.setup, self.crop = self.parent().setup, self.parent().crop
        self._ij, self._p = self._pattern_to_ij_and_q(pattern)
        self._idxs = np.arange(self._ij.size)

        _lbounds, _rbounds = [], []
        self._slices = [None, None, None]

        if bounds[0]:
            _lbounds += [-bounds[0],] * 3
            _rbounds += [bounds[0],] * 3
            self._slices[0] = slice(len(_lbounds) - 3, len(_lbounds))
        if bounds[1]:
            _lbounds += [1.0 - bounds[1],] * 3
            _rbounds += [1.0 + bounds[1],] * 3
            self._slices[1] = slice(len(_lbounds) - 3, len(_lbounds))
        if bounds[2]:
            _lbounds += [1.0 - bounds[2],] * 2
            _rbounds += [1.0 + bounds[2],] * 2
            self._slices[2] = slice(len(_lbounds) - 2, len(_lbounds))

        self._bounds = (_lbounds, _rbounds)
        self.x0 = np.mean(self._bounds, axis=0)
        self.alpha = self.alpha / (np.asarray(self._bounds[1]) - self.x0)
        self.update_hkl(self.x0, q_abs)

    def _get_slice(self, x: np.ndarray, i: int) -> np.ndarray:
        if self._slices[i]:
            return x[self._slices[i]]
        return (np.zeros(3), np.ones(3), np.ones(2))[i]

    def _pattern_to_ij_and_q(self, pattern: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        return pattern['x'] + self.shape[1] * pattern['y'], pattern['p']

    def generate_sample(self, x: np.ndarray) -> Sample:
        """Return a sample position and alignment.

        Args:
            x : Refinement solution.

        Returns:
            A new sample object.
        """
        return Sample(Rotation.import_euler(self._get_slice(x, 0)) * self.sample.rotation,
                      self.sample.position * self._get_slice(x, 1))

    def generate_setup(self, x: np.ndarray) -> ScanSetup:
        """Return an experimental setup.

        Args:
            x : Refinementa solution.

        Returns:
            A new experimental setup.
        """
        foc_pos = np.copy(self.setup.foc_pos)
        foc_pos[:2] *= self._get_slice(x, 2)
        return self.setup.replace(foc_pos=foc_pos)

    def generate_model(self, x: np.ndarray) -> CBDModel:
        """Return a CBD pattern prediction model, that provides an interface to generate a CBD
        pattern in different formats.

        Args:
            x : Refinement solution.

        Returns:
            A new CBD pattern prediction model.
        """
        return CBDModel(basis=self.basis, sample=self.generate_sample(x), transform=self.crop,
                        setup=self.generate_setup(x))

    def generate_streaks(self, x: np.ndarray) -> Streaks:
        """Generate a CBD pattern and return a set of predicted diffraction streaks.

        Args:
            x : Refinement solution.

        Returns:
            A set of predicted diffraction streaks.
        """
        return self.generate_model(x).generate_streaks(self.hkl, self.width)

    def pattern_dataframe(self, x: np.ndarray) -> pd.DataFrame:
        """Generate a CBD pattern in :class:`pandas.DataFrame` table format.

        Args:
            x : Refinement solution.

        Returns:
            A predicted CBD pattern.
        """
        return self.generate_model(x).pattern_dataframe(hkl=self.hkl, width=self.width,
                                                        dp=1.0 / self.q_max, profile='linear')

    def pattern_image(self, x: np.ndarray) -> np.ndarray:
        """Generate a CBD pattern as an image array.

        Args:
            x : Refinement solution.

        Returns:
            A predicted CBD pattern.
        """
        streaks = self.generate_streaks(x)
        return streaks.pattern_image(self.shape, dp=1.0 / self.q_max, profile='linear')

    def pattern_mask(self, x: np.ndarray) -> np.ndarray:
        """Generate a CBD pattern and return a mask where the streaks are located on the detector.

        Args:
            x : Refinement solution.

        Returns:
            A predicted CBD pattern mask.
        """
        streaks = self.generate_streaks(x)
        return streaks.pattern_mask(self.shape, max_val=self.q_max, profile='linear')

    def pattern_dict(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate a CBD pattern in dictionary format.

        Args:
            x : Refinement solution.

        Returns:
            A predicted CBD pattern.
        """
        streaks = self.generate_streaks(x)
        return streaks.pattern_dict(self.shape, dp=1.0 / self.q_max, profile='linear')

    def fitness(self, x: np.ndarray) -> List[float]:
        """Calculate the marginal log-likelihood, that the experimentally measured pattern
        corresponds to the predicted CBD pattern.

        Args:
            x : Refinement solution.

        Returns:
            Marginal log-likelihood.
        """
        criterion = cross_entropy(x=self._ij, p=self._p, q=self.pattern_mask(x).ravel(),
                                  q_max=self.q_max, epsilon=self.epsilon)
        return [criterion + np.sum(self.alpha * np.abs(x - self.x0)),]

    def get_bounds(self) -> Tuple[List[float], List[float]]:
        """Return lower and upper sample refinement bounds.

        Returns:
            A tuple of two sets of bounds, lower and upper respectively.
        """
        return self._bounds

    def update_hkl(self, x: np.ndarray, q_abs: float) -> None:
        """Update a set of reciprocal lattice points used in the prediction. The points are
        taken from inside the sphere of radius ``q_abs``, that fall into the Bragg condition.

        Args:
            x : Refinement solution.
            q_abs : Sphere radius. Reciprocal vectors are normalised and lie in [0.0 - 1.0]
                interval.
        """
        self.hkl = self.generate_model(x).generate_hkl(q_abs)

    def index_frame(self, x: np.ndarray, frame: int, iterations: int=4,
                    num_threads: int=1) -> pd.DataFrame:
        """Index the parent CBC table. Add miller indices to the pixel data in the table.
        Performs the binary dilation of the predicted CBD pattern to assign the Miller indices
        to the pixel data, that has no match with the predicted data.

        Args:
            x : Refinement solution.
            frame : Frame index.
            iterations : Number of binary dilations to perform.
            num_threads : Number of thread used in the calculations.

        Returns:
            A new CBC table of the given pattern in :class:`pandas.DataFrame` format.
        """
        if self.parent().crop != self.crop:
            raise ValueError('Parent Crop object has been changed, '\
                             'please create new SampleProblem instance.')
        if self.parent().setup != self.setup:
            raise ValueError('Parent ScanSetup object has been changed, '\
                             'please create new SampleProblem instance.')

        df_frame = self.parent().pattern_dataframe(frame)
        df_frame['ij'] = self._pattern_to_ij_and_q(df_frame)[0]
        df_frame = df_frame.sort_values('ij')

        streaks = self.generate_streaks(x)
        df = streaks.pattern_dataframe(self.shape, dp=1.0 / self.q_max, profile='linear')
        df = df.sort_values('p', ascending=True)
        df['ij'] = self._pattern_to_ij_and_q(df)[0]
        df = df[np.in1d(df['ij'], df_frame['ij'])]

        mask = np.zeros(self.shape, dtype=bool)
        labels = np.zeros(self.shape, dtype=np.uint32)
        mask[df_frame['y'], df_frame['x']] = True
        labels[df['y'], df['x']] = df['index']
        for _ in range(iterations):
            labels = median_filter(labels, footprint=self.footprint, inp_mask=labels,
                                   mask=mask, num_threads=num_threads)

        df_diff = df_frame[~np.in1d(df_frame['ij'], df['ij'])]
        ij_sim = np.where(labels.ravel())[0]
        ij_diff = ij_sim[~np.in1d(ij_sim, df['ij'])]
        df_idxs = df_diff.index[df_diff['ij'].searchsorted(ij_diff)]
        df_diff.loc[df_idxs, ['h', 'k', 'l']] = streaks.hkl[labels.ravel()[ij_diff], :]

        df_idxs = df_frame.index[df_frame['ij'].searchsorted(df['ij'])]
        df_inst = df_frame.loc[df_idxs]
        df_inst.loc[:, ['h', 'k', 'l']] = df[['h', 'k', 'l']].values
        return pd.concat((df_inst, df_diff)).drop('ij', axis=1)

@dataclass
class IntensityScaler(DataContainer):
    """Iterative CBC dataset intensity scaling algorithm. Provides an interface
    to generate a crystal diffraction efficiency mapping (xtal) and crystal structure
    factors (sfac).

    Args:
        fidxs : Array of CBC table first row indices pertaining to different CBC patterns.
        frames : Array of unique frame indices stored in the table.
        hkl : Array of unique Miller indices stored inside the CBC table.
        hkl_idxs : Array of output hkl list indices.
        iidxs : Array of CBC table first row indices pertaining to different CBC streaks.
        num_threads : Number of threads used in the calculations.
        parent : Reference to the parent CBC table.
        sfac : Crystal structure factors.
        sfac_err : Crystal structure factor uncertainties.
        signal : Diffraction signal.
        xidx : Set of crystal diffraction map frame indices.
        xtal : Crystal diffraction efficiency mapping.
        xtal_map : Mapping of diffraction signal into the crystal plane grid.
    """
    parent      : ReferenceType[CBCTable]
    frames      : np.ndarray
    fidxs       : np.ndarray
    iidxs       : np.ndarray
    hkl         : np.ndarray
    hkl_idxs    : np.ndarray
    xtal_map    : np.ndarray
    signal      : np.ndarray
    num_threads : int

    xidx        : Optional[np.ndarray] = None
    xtal        : Optional[np.ndarray] = None
    sfac        : Optional[np.ndarray] = None
    sfac_err    : Optional[np.ndarray] = None

    step        : ClassVar[np.ndarray] = np.ones(2)

    def __post_init__(self):
        if self.xidx is None:
            frames = self.parent().table['frames']
            self.xidx = frames.map({f: idx for idx, f in enumerate(self.frames)}).to_numpy()

    def criterion(self) -> float:
        r"""Return the mean abolute error (MAE) of the intensity scaling problem.

        Notes:
            The MAE is given by:

            .. math::
                L(I, D_{xtal}, F_{xtal}) = \frac{1}{N} \sum_{i = 0}^N \left| I - D_{xtal}(x_i, y_i)
                F_{xtal}(hkl_i) \right|,

            where :math:`I` - diffraction signal, :math:`D_{xtal}` - crystal diffraction efficiency
            map, and :math:`F_{xtal}` - crystal structure factors.

        Returns:
            Mean absolute error.
        """
        if self.sfac is None:
            raise AttributeError("'sfac' attribute is not defined")
        if self.xtal is None:
            raise AttributeError("'xtal' attribute is not defined")
        return scaling_criterion(sf=self.sfac, sgn=self.signal, xidx=self.xidx,
                                 xmap=self.xtal_map, xtal=self.xtal, iidxs=self.iidxs,
                                 num_threads=self.num_threads)

    def export_sfac(self, path: str):
        """Export structure factors to a text hkl file.

        Args:
            path : Path to the output file.
        """
        if self.sfac is None:
            raise AttributeError("'sfac' attribute is not defined")
        sfac = np.empty(self.hkl.shape[0])
        sfac[self.hkl_idxs] = self.sfac[self.iidxs[:-1]]
        sfac_err = np.empty(self.hkl.shape[0])
        sfac_err[self.hkl_idxs] = self.sfac_err[self.iidxs[:-1]]
        norm = sfac_err.min()
        idxs = np.lexsort(np.abs(self.hkl.T)[::-1])
        np.savetxt(path, np.concatenate((self.hkl[idxs], sfac[idxs, None] / norm,
                                         sfac_err[idxs, None] / norm), axis=1),
                   ' %4d %4d %4d %10.2f %10.2f')

    def update_xtal(self, bandwidth: float) -> IntensityScaler:
        """Generate a new crystal map using the kernel regression.

        Args:
            bandwidth : kernel bandwidth in pixels.

        Returns:
            New :class:`IntensityScaler` container with the updated crystal efficiency map.
        """
        if self.sfac is None:
            raise AttributeError("'sfac' attribute is not defined")
        pupils = []
        for f0, f1 in zip(self.fidxs, self.fidxs[1:]):
            pupils.append(kr_grid(y=self.signal[f0:f1] / self.sfac[f0:f1], x=self.xtal_map[f0:f1],
                                  step=self.step, sigma=bandwidth, cutoff=3.0 * bandwidth,
                                  num_threads=self.num_threads))
        norm = np.mean(pupils)
        return self.replace(xtal=np.stack(pupils) / norm, sfac=self.sfac * norm,
                            sfac_err=self.sfac_err * norm)

    def update_sfac(self) -> IntensityScaler:
        """Generate new crystal structure factors.

        Returns:
            New :class:`IntensityScaler` container with the updated crystal structure factors.
        """
        if self.xtal is None:
            raise AttributeError("'xtal' attribute is not defined")
        sfac, sfac_err = update_sf(sgn=self.signal, xidx=self.xidx, xmap=self.xtal_map,
                                   xtal=self.xtal, hkl_idxs=self.hkl_idxs, iidxs=self.iidxs,
                                   num_threads=self.num_threads)
        return self.replace(sfac=sfac, sfac_err=sfac_err)

    def update_table(self):
        """Update the parent :class:`cbclib.CBCTable`.
        """
        if self.sfac is None:
            raise AttributeError("'sfac' attribute is not defined")
        if self.xtal is None:
            raise AttributeError("'xtal' attribute is not defined")
        self.parent().table['sfac'] = self.sfac
        self.parent().table['xtal'] = xtal_interpolate(xidx=self.xidx, xmap=self.xtal_map,
                                                       xtal=self.xtal, num_threads=self.num_threads)

    def train(self, bandwidth: float, n_iter: int=10, f_tol: float=1e-3,
              return_extra: bool=False, verbose: bool=True) -> IntensityScaler:
        """Perform an iterative update of crystal diffraction efficiency map (xtal) and crystal
        structure factors (sfac) until the mean absolute error converges to a minimum. The kernel
        bandwidth in the diffraction efficiency map update is held fixed during the iterative
        update.

        Args:
            bandwidth : Kernel bandwidth used in the diffraction efficiency map (xtal) update.
            n_iter : Maximum number of iterations.
            f_tol : Tolerance for termination by the change of the average error. The
                iteration stops when :math:`(f^k - f^{k + 1}) / max(|f^k|, |f^{k + 1}|) <= f_{tol}`.
            return_extra : Return errors at each iteration if True.
            verbose : Set verbosity of the computation process.

        Returns:
            A tuple of two items (`scaler`, `errors`). The elements of the tuple
            are as follows:

            * `scaler` : A new :class:`IntensityScaler` object with the updated
              ``xtal`` and ``sfac``.
            * `errors` : List of mean absolute errors for each iteration. Only if
              ``return_extra`` is True.
        """
        shape = self.xtal_map.max(axis=0).astype(int) + 1
        xtal = np.ones((self.frames.size, shape[0], shape[1]), dtype=np.float32)
        obj = self.replace(xtal=xtal).update_sfac()

        itor = tqdm(range(1, n_iter + 1), disable=not verbose,
                    bar_format='{desc} {percentage:3.0f}% {bar} '\
                    'Iteration {n_fmt} / {total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')

        errors = [obj.criterion()]
        if verbose:
            itor.set_description(f"Error = {errors[-1]:.6f}")

        for _ in itor:
            new_obj = obj.update_xtal(bandwidth=bandwidth)
            new_obj = new_obj.update_sfac()
            errors.append(new_obj.criterion())

            if verbose:
                itor.set_description(f"Error = {errors[-1]:.6f}")

            if (errors[-2] - errors[-1]) / max(errors[-2], errors[-1]) > f_tol:
                obj = new_obj
            else:
                break

        if return_extra:
            return obj, errors
        return obj
