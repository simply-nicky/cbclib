from __future__ import annotations
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple
from dataclasses import InitVar, dataclass
from weakref import ref
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm.auto import tqdm

from .bin import (euler_matrix, draw_line_image, ce_criterion, unique_indices, kr_grid,
                  poisson_criterion, ls_criterion, unmerge_signal)
from .cbc_indexing import Map3D, FourierIndexer
from .cbc_setup import Basis, Rotation, Sample, ScanSamples, ScanSetup, Streaks, CBDModel
from .cxi_protocol import Indices
from .data_container import DataContainer, Crop, ReferenceType

@dataclass
class CBCTable():
    """Convergent beam crystallography tabular data. The data is stored in :class:`pandas.DataFrame`
    table. A table must contain the following columns:

    * `frames` : Frame index.
    * `index` : Diffraction streak index.
    * `x`, `y` : x and y pixel coordinate.
    * `p` : Normalised pattern value. The value lies in (0.0 - 1.0) interval.
    * `rp` : Reflection profiles.
    * `I_raw` : Raw photon count.
    * `bgd` : Background intensity.
    * `h`, `k`, `l` : Miller indices.
    * `sfac` : Crystal structure factors.
    * `xtal` : Crystal diffraction efficiencies.

    Args:
        table : CBC tabular data.
        setup : Experimental setup.
        crop : Detector region of interest.
    """
    columns         : ClassVar[Set[str]] = {'frames', 'index', 'x', 'y', 'p', 'rp', 'I_raw', 'bgd'}
    table           : pd.DataFrame
    setup           : ScanSetup
    crop            : Optional[Crop] = None

    def __post_init__(self):
        if not self.columns.issubset(self.table.columns):
            raise ValueError(f'Dataframe must contain the following columns: {self.columns}')
        if self.crop is None:
            self.crop = self.get_crop()
        self.frames, self.fidxs, self.finv = unique_indices(self.table['frames'].to_numpy())

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.frames.size, self.crop.roi[1] - self.crop.roi[0], self.crop.roi[3] - self.crop.roi[2])

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
            path : Path to the HDF5 file.
            key : The group identifier in the HDF5 file.
            setup : Experimental geometry.

        Returns:
            A new CBC table object.
        """
        return cls(pd.read_hdf(path, key, usecols=cls.columns), setup)

    @property
    def dtype(self) -> np.dtype:
        return self.table.dtypes['bgd']

    def __getitem__(self, idxs: Indices) -> CBCTable:
        return CBCTable(self.table.loc[idxs], self.setup, self.crop)

    def _repr_html_(self) -> Optional[str]:
        return self.table._repr_html_()

    def fourier_index(self, samples: ScanSamples, qx_arr: np.ndarray, qy_arr: np.ndarray,
                      qz_arr: np.ndarray, num_threads: int=1) -> FourierIndexer:
        """Map the measured normalised intensities to the reciprocal space. Returns a
        :class:`cbclib.Map3D` 3D data container capable of performing the auto Fourier indexing.

        Args:
            samples : Set of scan samples.
            qx_arr : Array of reciprocal x coordinates.
            qy_arr : Array of reciprocal y coordinates.
            qz_arr : Array of reciprocal z coordinates.
            num_threads : Number of threads used in the calculations.

        Returns:
            3D data container of measured normalised intensities.
        """
        q_map = np.zeros((qz_arr.size, qy_arr.size, qx_arr.size), dtype=self.dtype)
        kout = samples.detector_to_kout(self.table['x'].to_numpy(), self.table['y'].to_numpy(),
                                        self.setup, self.finv, num_threads=num_threads)
        rec_vec = kout - self.setup.kin_center
        rec_vec = samples.rotate(rec_vec, self.finv, reciprocate=True, num_threads=num_threads)
        x_idxs = np.searchsorted(qx_arr, rec_vec[:, 0])
        y_idxs = np.searchsorted(qy_arr, rec_vec[:, 1])
        z_idxs = np.searchsorted(qz_arr, rec_vec[:, 2])
        mask = (x_idxs < qx_arr.size) & (y_idxs < qy_arr.size) & (z_idxs < qz_arr.size)
        np.add.at(q_map, (z_idxs[mask], y_idxs[mask], x_idxs[mask]),
                  self.table['p'].to_numpy()[mask])
        return FourierIndexer(val=q_map, x=qx_arr, y=qy_arr, z=qz_arr, num_threads=num_threads)

    def drop_duplicates(self, method: str='keep_best') -> pd.DataFrame:
        """Discard the pixel data, that has duplicate `x`, `y` coordinates.

        Args:
            method : Choose the policy of dealing with the pixel data that has duplicate x and
                y coordinates:

                * `keep_all` : Keep duplicated data.
                * `keep_best` : Keep the pixels that have higher likelihood value `p`.
                * `ignore` : Discard duplicated data.

        Returns:
            New CBC table with the duplicate data discarded.
        """
        if method == 'keep_all':
            table = self.table
        elif method == 'keep_best':
            mask = self.table.duplicated(['frames', 'x', 'y'], keep=False)
            duplicates = self.table[mask].sort_values(['frames', 'x', 'y', 'rp'])
            duplicates = duplicates.drop_duplicates(['frames', 'x', 'y'])
            table = pd.concat((self.table[~mask], duplicates)).sort_index()
        elif method == 'ignore':
            table = self.table.drop_duplicates(['frames', 'x', 'y'], keep=False)
        else:
            raise ValueError('Invalid duplicates keyword')

        return CBCTable(table.reset_index(drop=True), self.setup, self.crop)

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

        rec_vec = self.table.loc[:, ['h', 'k', 'l']].to_numpy().dot(basis.mat)
        rec_vec = samples.rotate(rec_vec, self.finv, num_threads=num_threads)
        kout = samples.detector_to_kout(self.table['x'].to_numpy(), self.table['y'].to_numpy(),
                                        self.setup, self.finv, rec_vec, num_threads)
        return kout - rec_vec

    def get_crop(self) -> Crop:
        """Return the region of interest on the detector plane.

        Returns:
            A new crop object with the ROI, inferred from the CBC table.
        """
        return Crop((self.table['y'].min(), self.table['y'].max() + 1,
                     self.table['x'].min(), self.table['x'].max() + 1))

    def get_frames(self, frames: Indices) -> CBCTable:
        """Return a subset of frames ``frames``.

        Args:
            frames : Frame indices.

        Returns:
            A new :class:`CBCTable` with a subset of frames.
        """
        dataframes = []
        for frame in np.sort(np.atleast_1d(frames)):
            index = self.frames.tolist().index(frame)
            dataframes.append(self.table[self.fidxs[index]:self.fidxs[index + 1]])
        return CBCTable(pd.concat(dataframes, ignore_index=True), self.setup, self.crop)

    def pattern_dataframe(self, frame: int) -> pd.DataFrame:
        """Return a single pattern table. The `x`, `y` coordinates are transformed by the ``crop``
        attribute.

        Args:
            frame : Frame index.

        Returns:
            A :class:`pandas.DataFrame` table.
        """
        index = self.frames.tolist().index(frame)
        df_frame = self.table[self.fidxs[index]:self.fidxs[index + 1]]
        mask = (self.crop.roi[0] < df_frame['y']) & (df_frame['y'] < self.crop.roi[1]) & \
               (self.crop.roi[2] < df_frame['x']) & (df_frame['x'] < self.crop.roi[3])
        df_frame = df_frame[mask]
        pts = np.stack(self.crop.forward_points(df_frame['x'], df_frame['y']), axis=-1)
        df_frame.loc[:, ['x', 'y']] = pts
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

    def pattern_image(self, frame: int, key: str='rp') -> np.ndarray:
        """Return a CBC pattern image array of the given attribute `key`. The `x`, `y` coordinates
        are transformed by the ``crop`` attribute.

        Args:
            frame : Frame index.
            key : Attribute's name.

        Returns:
            A pattern image array.
        """
        df_frame = self.pattern_dataframe(frame)
        pattern = np.zeros((self.crop.roi[1] - self.crop.roi[0],
                            self.crop.roi[3] - self.crop.roi[2]), dtype=self.dtype)
        pattern[df_frame['y'], df_frame['x']] = df_frame[key]
        return pattern

    def refine_samples(self, bounds: np.ndarray, basis: Basis, hkl: np.ndarray,
                       samples: ScanSamples, width: float, alpha: float=0.0) -> SampleRefiner:
        """Return a :class:`SampleRefiner` object designed to perform the sample refinement.
        The sample refinement yields a set of sample parameters, that attain the best fit between
        simulated and experimentally measured patterns. The refinement is performed for each pattern
        separately.

        Args:
            bounds : Bounds for sample refinement variables. A set of ``(min, max)`` pairs of the
                shape ``(N, 2)``, where ``N`` is the number of variables. Each of the terms is
                diregarded in the refinement process, if the corresponding bounds are equal to 0.
            basis : Basis vectors of crystal lattice unit cell.
            hkl : Array of Miller indices used in the refinement.
            samples : Sample object. The object is given by the rotation matrix and a sample
                position.
            width : Diffraction streak width in pixels. The value is used to generate a predicted
                CBC pattern.
            alpha : Regularisation term in the loss function.

        Returns:
            A CBC sample refinement object.

        See Also:
            cbclib.SampleRefiner : CBC sample refiner class.
        """
        samples = ScanSamples({frame: samples[frame] for frame in self.frames})
        smp_ds = self.setup.smp_dist * np.ones(len(samples))
        return SampleRefiner(bounds=bounds, rmats=samples.rmats, smp_ds=smp_ds, basis=basis,
                             hkl=hkl, frames=self.frames, fidxs=self.fidxs, width=width,
                             parent=ref(self), alpha=alpha)

    def refine_setup(self, bounds: np.ndarray, basis: Basis, hkl: np.ndarray, tilts: np.ndarray,
                     width: float, alpha: float=0.0) -> SetupRefiner:
        """Return a :class:`SetupRefiner` object designed to perform the setup refinement.
        Setup refinement yields a scattering geometry parameters, that attain the best fit between
        simulated and experimentally measured patterns. The refinement is performed for the whole
        scan in one go.

        Args:
            bounds : Bounds for setup refinement variables. A set of ``(min, max)`` pairs of the
                shape ``(N, 2)``, where ``N`` is the number of variables. Each of the terms is
                diregarded in the refinement process, if the corresponding bounds are equal to 0.
            basis : Basis vectors of crystal lattice unit cell.
            sample : Sample object. The object is given by the rotation matrix and a sample
                position.
            q_abs : Size of the recpirocal space. Reciprocal vectors are normalised, and span in
                [0.0 - 1.0] interval.
            width : Diffraction streak width in pixels. The value is used to generate a predicted
                CBC pattern.
            alpha : Regularisation term in the loss function.

        Returns:
            A CBC setup refinement object.

        See Also:
            cbclib.SetupRefiner : CBC setup refinement class.
        """
        return SetupRefiner(bounds=bounds, tilts=tilts, basis=basis, frames=self.frames, hkl=hkl,
                            fidxs=self.fidxs, width=width, parent=ref(self), alpha=alpha)

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

        kins = self.generate_kins(basis, samples, num_threads)
        kin_min, kin_max = kins.min(axis=0), kins.max(axis=0)

        xtal = Map3D(val=np.ones((self.shape[0], xtal_shape[1], xtal_shape[0])),
                     x=np.linspace(kin_min[0], kin_max[0], xtal_shape[0], endpoint=True),
                     y=np.linspace(kin_min[1], kin_max[1], xtal_shape[1], endpoint=True),
                     z=self.frames, num_threads=num_threads)
        xmap = np.concatenate((kins[:, :2], self.table['frames'].to_numpy()[:, None]), axis=1)
        return IntensityScaler(parent=ref(self), frames=self.frames, fidxs=self.fidxs, xtal=xtal,
                               xmap=xmap, num_threads=num_threads)

    def update_crop(self, crop: Optional[Crop]=None) -> CBCTable:
        """Return a new CBC table with the updated region of interest.

        Args:
            crop : A new region of interest.

        Returns:
            A new CBC table with the updated ROI.
        """
        return CBCTable(self.table, self.setup, crop)

class Refiner():
    parent    : ReferenceType[CBCTable]
    basis     : Basis
    fidxs     : np.ndarray
    frames    : np.ndarray
    hkl       : np.ndarray
    width     : float
    alpha     : float = 0.0

    profile   : ClassVar[str] = 'linear'
    epsilon   : ClassVar[float] = 1e-12

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.frames.size, self.crop.roi[1] - self.crop.roi[0], self.crop.roi[3] - self.crop.roi[2])

    def __getstate__(self) -> Dict[str, Any]:
        state = {key: val for key, val in self.__dict__.items() if key != 'parent'}
        state['parent'] = None
        return state

    def __post_init__(self, bounds: np.ndarray):
        self.setup, self.crop = self.parent().setup, self.parent().crop
        x, y = self.crop.forward_points(self.parent().table['x'].to_numpy(dtype=np.uint32),
                                        self.parent().table['y'].to_numpy(dtype=np.uint32))
        self.ij = x + self.shape[2] * y
        self.p = self.parent().table['p'].to_numpy(dtype=np.float32)
        self._idxs = np.asarray(bounds[1] - bounds[0], dtype=bool)
        self._bounds = bounds[:, self._idxs]
        self._slices = self.generate_slices(self.frames)

    @property
    def x0(self) -> np.ndarray:
        return np.mean(self.get_bounds(), axis=0)

    @staticmethod
    def generate_slices(frames: Indices) -> Dict[str, slice]:
        raise NotImplementedError

    def x_ext(self, x: np.ndarray) -> np.ndarray:
        x_ext = np.zeros(self._idxs.size)
        np.place(x_ext, self._idxs, x)
        return x_ext

    def generate_samples(self, x: np.ndarray) -> ScanSamples:
        raise NotImplementedError

    def generate_setup(self, x: np.ndarray) -> ScanSetup:
        raise NotImplementedError

    def generate_basis(self, x: np.ndarray) -> Basis:
        raise NotImplementedError

    def generate_models(self, x: np.ndarray) -> Dict[int, CBDModel]:
        """Return a CBD pattern prediction model, that provides an interface to generate a CBD
        pattern in different formats.

        Args:
            x : Refinement solution.

        Returns:
            A new CBD pattern prediction model.
        """
        basis, setup = self.generate_basis(x), self.generate_setup(x)
        models = {idx: CBDModel(basis=basis, sample=sample, transform=self.crop, setup=setup)
                  for idx, sample in enumerate(self.generate_samples(x).values())}
        return models

    def generate_streaks(self, x: np.ndarray) -> Dict[int, Streaks]:
        """Generate a CBD pattern and return a set of predicted diffraction streaks.

        Args:
            x : Refinement solution.

        Returns:
            A set of predicted diffraction streaks.
        """
        return {idx: model.generate_streaks(self.hkl, self.width)
                for idx, model in self.generate_models(x).items()}

    def patterns_dataframe(self, x: np.ndarray) -> pd.DataFrame:
        """Generate a CBD pattern in :class:`pandas.DataFrame` table format.

        Args:
            x : Refinement solution.

        Returns:
            A predicted CBD pattern.
        """
        dataframes = []
        for idx, model in self.generate_models(x).items():
            df = model.pattern_dataframe(hkl=self.hkl, width=self.width, profile=self.profile)
            df['frames'] = self.frames[idx]
            dataframes.append(df)
        return pd.concat(dataframes, ignore_index=True)

    def patterns_image(self, x: np.ndarray) -> np.ndarray:
        """Generate a CBD pattern and return a mask where the streaks are located on the detector.

        Args:
            x : Refinement solution.

        Returns:
            A predicted CBD pattern mask.
        """
        lines = [streaks.to_lines() for streaks in self.generate_streaks(x).values()]
        return draw_line_image(self.shape, lines=lines, profile=self.profile)

    def fitness(self, x: np.ndarray, num_threads: int=1) -> List[float]:
        """Calculate the marginal log-likelihood, that the experimentally measured pattern
        corresponds to the predicted CBD pattern.

        Args:
            x : Refinement solution.

        Returns:
            Marginal log-likelihood.
        """
        lines = [streaks.to_lines() for streaks in self.generate_streaks(x).values()]
        criterion = ce_criterion(ij=self.ij, p=self.p, fidxs=self.fidxs, shape=self.shape[1:],
                                 lines=lines, epsilon=self.epsilon, profile=self.profile,
                                 num_threads=num_threads)
        if self.alpha:
            criterion += np.sum(self.alpha * np.abs(x - self.x0))
        return [criterion,]

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return lower and upper sample refinement bounds.

        Returns:
            A tuple of two sets of bounds, lower and upper respectively.
        """
        return self._bounds

    def filter_hkl(self, x: np.ndarray):
        """Update a set of reciprocal lattice points used in the prediction. The points are
        taken from inside the sphere of radius ``q_abs``, that fall into the Bragg condition.

        Args:
            x : Refinement solution.
        """
        idxs = np.concatenate([np.arange(self.hkl.shape[0])[model.filter_hkl(self.hkl)]
                               for model in self.generate_models(x).values()])
        self.hkl = self.hkl[np.unique(idxs)]

@dataclass
class SampleRefiner(Refiner):
    """Sample refinement class. It employs :class:`cbclib.CBDModel` CBD pattern prediction
    to find how well the current estimate of sample positions and alignments fits with the
    experimentally measured patterns. The class provides a method to calculate the fitness
    criterion, which is based on cross-entropy between the experimentally measured and
    predicted patterns.

    Args:
        parent : Reference to the parent CBC table.
        bounds : Bounds for sample refinement variables. A set of ``(min, max)`` pairs of the
            shape ``(N, 2)``, where ``N`` is the number of variables. Each of the terms is
            diregarded in the refinement process, if the corresponding bounds are equal to 0.
        basis : Basis vectors of lattice unit cell.
        fidxs : Array of first indices pertaining to different patterns.
        frames : Frame indices of the measured patterns.
        hkl : Set of reciprocal lattice points used in the prediction.
        rmats : A set of sample's rotation matrices.
        smp_ds : A set of focus-to-sample distances [m].
        width : Diffraction streak width in pixels. The value is used to generate a predicted
            CBC pattern.
        alpha : Regularisation term in the loss function.
    """
    parent    : ReferenceType[CBCTable]
    bounds    : InitVar[np.ndarray]
    rmats     : np.ndarray
    smp_ds    : np.ndarray
    basis     : Basis
    fidxs     : np.ndarray
    frames    : np.ndarray
    hkl       : np.ndarray
    width     : float
    alpha     : float = 0.0

    @staticmethod
    def generate_slices(frames: Indices) -> Dict[str, slice]:
        size = np.array(frames).size
        return {'smp_d': slice(0, size), 'tilt': slice(size, 4 * size)}

    @classmethod
    def generate_bounds(cls, z_tol: float, tilt_tol: float, frames: Indices,
                        x0: Optional[np.ndarray]=None) -> np.ndarray:
        """Return a set of bounds for sample refinement variables based on the given set of
        tolerances.

        Args:
            z_tol : Tolerance of focus-to-sample distance [0.0 - 1.0].
            tilt_tol : Tolerance of sample tilt angles in radians.
            frames : A set of frames indices.
            x0 : Initial set of refinement variables.

        Returns:
            A set of ``(min, max)`` pairs for sample refinement variables.
        """
        slices = cls.generate_slices(frames)
        ub = np.concatenate((z_tol * np.ones(slices['smp_d'].stop - slices['smp_d'].start),
                             tilt_tol * np.ones(slices['tilt'].stop - slices['tilt'].start)))
        x_init = np.zeros(ub.size)
        if x0 is not None:
            idxs = np.arange(ub.size)[ub.astype(bool)]
            x_init = np.zeros(ub.size)
            x_init[idxs] = x0
        return np.stack((x_init - ub, x_init + ub))

    def generate_basis(self, x: np.ndarray) -> Basis:
        """Return an experimental setup.

        Args:
            x : Refinement solution.

        Returns:
            A new experimental setup.
        """
        return self.basis

    def generate_samples(self, x: np.ndarray) -> ScanSamples:
        """Return a sample position and alignment.

        Args:
            x : Refinement solution.

        Returns:
            A new sample object.
        """
        foc_pos = self.generate_setup(x).foc_pos
        angles = self.x_ext(x)[self._slices['tilt']].reshape(-1, 3)
        rmats = np.matmul(self.rmats, euler_matrix(angles))
        smp_ds = self.smp_ds * (1.0 + self.x_ext(x)[self._slices['smp_d']])
        return ScanSamples({frame: Sample(Rotation(rmats[idx]), foc_pos[2] + smp_ds[idx])
                            for idx, frame in enumerate(self.frames)})

    def generate_setup(self, x: np.ndarray) -> ScanSetup:
        """Return an experimental setup.

        Args:
            x : Refinement solution.

        Returns:
            A new experimental setup.
        """
        return self.setup

@dataclass
class SetupRefiner(Refiner):
    """Setup refinement class. It employs :class:`cbclib.CBDModel` CBD pattern prediction
    to find how well the current estimate of scattering geometry parameters fits with the
    experimentally measured patterns. The class provides a method to calculate the fitness
    criterion, which is based on cross-entropy between the experimentally measured and
    predicted patterns.

    Args:
        parent : Reference to the parent CBC table.
        bounds : Bounds for setup refinement variables. A set of ``(min, max)`` pairs of the
            shape ``(N, 2)``, where ``N`` is the number of variables. Each of the terms is
            diregarded in the refinement process, if the corresponding bounds are equal to 0.
        basis : Basis vectors of lattice unit cell.
        fidxs : Array of first indices pertaining to different patterns.
        frames : Frame indices of the measured patterns.
        hkl : Set of reciprocal lattice points used in the prediction.
        tilts : A set of tilt angles of sample rotation.
        width : Diffraction streak width in pixels. The value is used to generate a predicted
            CBC pattern.
        alpha : Regularisation term in the loss function.
    """
    parent    : ReferenceType[CBCTable]
    bounds    : InitVar[np.ndarray]
    basis     : Basis
    fidxs     : np.ndarray
    frames    : np.ndarray
    hkl       : np.ndarray
    tilts     : np.ndarray
    width     : float
    alpha     : float = 0.0

    @staticmethod
    def generate_slices(frames: Indices) -> Dict[str, slice]:
        tilt_size = np.array(frames, dtype=bool).sum()
        return {'lat_d': slice(0, 3), 'lat_a': slice(3, 9), 'f_pos': slice(9, 12),
                'rot_a': slice(12, 14), 'smp_d': slice(14, 15), 'tilt': slice(15, 15 + tilt_size)}

    @classmethod
    def generate_bounds(cls, lat_tol: Tuple[float, float], foc_tol: float, rot_tol: float,
                        z_tol: float, tilt_tol: float, frames: Indices,
                        x0: Optional[np.ndarray]=None) -> np.ndarray:
        """Return a set of bounds for setup refinement variables based on the given set of
        tolerances.

        Args:
            lat_tol : A tuple of ``(d_tol, ang_tol)`` tolerances, where ``d_tol`` is the tolerance
                of lattice constants [0.0 - 1.0] and ``ang_tol`` is the tolerance of lattice angles
                in radians.
            foc_tol : Tolerance of the focal point [0.0 - 1.0].
            rot_tol : Tolerance of the sample rotation axis in radians. The rotation axis is defined
                by azymuth and inclination.
            z_tol : Tolerance of focus-to-sample distance [0.0 - 1.0].
            tilt_tol : Tolerance of sample tilt angles.
            frames : A set of frames indices.
            x0 : Initial set of refinement variables.

        Returns:
            A set of ``(min, max)`` pairs for setup refinement variables.
        """
        slices = cls.generate_slices(frames)
        ub = np.concatenate((lat_tol[0] * np.ones(slices['lat_d'].stop - slices['lat_d'].start),
                             lat_tol[1] * np.ones(slices['lat_a'].stop - slices['lat_a'].start),
                             foc_tol * np.ones(slices['f_pos'].stop - slices['f_pos'].start),
                             rot_tol * np.ones(slices['rot_a'].stop - slices['rot_a'].start),
                             z_tol * np.ones(slices['smp_d'].stop - slices['smp_d'].start),
                             tilt_tol * np.ones(slices['tilt'].stop - slices['tilt'].start)))
        x_init = np.zeros(ub.size)
        if x0 is not None:
            idxs = np.arange(ub.size)[ub.astype(bool)]
            x_init = np.zeros(ub.size)
            x_init[idxs] = x0
        return np.stack((x_init - ub, x_init + ub))

    def generate_basis(self, x: np.ndarray) -> Basis:
        """Return an experimental setup.

        Args:
            x : Refinementa solution.

        Returns:
            A new experimental setup.
        """
        mat = self.basis.to_spherical().ravel()
        mat[::3] *= 1.0 + self.x_ext(x)[self._slices['lat_d']]
        mat[[1, 2, 4, 5, 7, 8]] += self.x_ext(x)[self._slices['lat_a']]
        return Basis.import_spherical(mat.reshape((3, 3)))

    def generate_samples(self, x: np.ndarray) -> ScanSamples:
        """Return a sample position and alignment.

        Args:
            x : Refinement solution.

        Returns:
            A new sample object.
        """
        thetas = np.zeros(self.shape[0])
        tilts = self.x_ext(x)[self._slices['tilt']]
        if tilts.size:
            thetas[-tilts.size:] += self.tilts[-tilts.size:] + tilts
        return self.generate_setup(x).tilt_samples(self.frames, thetas)

    def generate_setup(self, x: np.ndarray) -> ScanSetup:
        """Return an experimental setup.

        Args:
            x : Refinement solution.

        Returns:
            A new experimental setup.
        """
        foc_pos = self.setup.foc_pos * (1.0 + self.x_ext(x)[self._slices['f_pos']])
        rot_axis = self.setup.rot_axis + self.x_ext(x)[self._slices['rot_a']]
        smp_dist = self.setup.smp_dist * (1.0 + self.x_ext(x)[self._slices['smp_d']].item())
        return self.setup.replace(foc_pos=foc_pos, rot_axis=rot_axis, smp_dist=smp_dist)

@dataclass
class IntensityScaler(DataContainer):
    """Iterative CBC intensity scaling algorithm. Provides an interface to iteratively
    update crystal projection maps (xtal) and crystal structure factors (sfac) by
    fitting the modelled intensity profile of Bragg reflections to the experimentally
    measured intensities.

    Args:
        bgd : Background levels.
        fidxs : Array of CBC table first row indices pertaining to different CBC patterns.
        frames : Array of unique frame indices stored in the table.
        hkl : Array of unique Miller indices stored inside the CBC table.
        hkl_idxs : Array of Bragg reflection indices.
        I_raw : Experimentally measured intensities.
        idxs : Array of streak indices.
        num_threads : Number of threads used in the calculations.
        parent : Reference to the parent CBC table.
        prof : Standard reflection profiles.
        xmap : A set of aperture function coordinates, that correspond to the detector
            coordinates of experimentally measured intensities.
        xtal : Crystal projection maps.
        xtal_bi : Array of interpolated crystal diffraction power values at ``xmap``
            coordinates.
    """
    parent      : ReferenceType[CBCTable]
    frames      : np.ndarray
    fidxs       : np.ndarray
    xmap        : np.ndarray
    xtal        : Map3D
    num_threads : int

    bgd         : Optional[np.ndarray] = None
    I_raw       : Optional[np.ndarray] = None
    idxs        : Optional[np.ndarray] = None
    iidxs       : Optional[np.ndarray] = None
    ij          : Optional[np.ndarray] = None
    hkl         : Optional[np.ndarray] = None
    hkl_idxs    : Optional[np.ndarray] = None
    prof        : Optional[np.ndarray] = None
    xtal_bi     : Optional[np.ndarray] = None

    def __post_init__(self):
        if self.bgd is None:
            self.bgd = self.parent().table['bgd'].to_numpy(dtype=np.float32)
        if self.I_raw is None:
            self.I_raw = self.parent().table['I_raw'].to_numpy(dtype=np.uint32)
        if self.idxs is None or self.iidxs is None:
            self.iidxs, self.idxs = unique_indices(self.parent().table['index'].to_numpy())[1:]
        if self.ij is None:
            x, y = self.parent().crop.forward_points(self.parent().table['x'].to_numpy(),
                                                     self.parent().table['y'].to_numpy())
            self.ij = np.asarray(x + self.parent().shape[2] * y, dtype=np.uint32)
        if self.hkl is None or self.hkl_idxs is None:
            self.hkl = self.parent().table.loc[self.iidxs[:-1], ['h', 'k', 'l']].to_numpy()
            self.hkl_idxs = np.arange(self.hkl.shape[0], dtype=np.uint32)
        if self.prof is None:
            self.prof = self.parent().table['rp'].to_numpy(dtype=np.float32)
        if self.xtal_bi is None:
            self.xtal_bi = self.xtal.interpolate(self.xmap)

    def init_estimate(self) -> np.ndarray:
        """Return initial estimate of crystal structure factors and intercept values.

        Returns:
            Array of initial crystal structure factors and intercepts.
        """
        return np.zeros(self.idxs[-1] + self.hkl.shape[0] + 1, dtype=np.float32)

    def fitness(self, x: np.ndarray, fit_intercept: bool=True, kind: str='poisson',
                loss: str='l2') -> Tuple[float, np.ndarray]:
        r"""Return the fitness criterion and the jacobian matrix. The structure factors
        and projection maps are used to model the intensity profiles of Bragg reflections.
        Either negative log likelihood or least squares error is calculated to estimate
        how well the modelled intensity profiles fit to the experimental intensities.

        Args:
            x : Current estimate of crystal structure factors and intercept values.
            fit_intercept : Use intercept if True.
            kind : Choose between the negative log likelihood Poisson criterion ('poisson')
                and the least squares criterion ('least_squares')
            loss : Loss function to use in the least squares criterion. The following
                keyword arguments are allowed:

                * `l1`: L1 loss (absolute) function.
                * `l2` : L2 loss (squared) function.
                * `Huber` : Huber loss function.

        Notes:
            The modelled diffraction pattern :math:`\hat{I}_n(\mathbf{x}_i)` is given by:

            .. math::
               \hat{I}_n(\mathbf{x}_i) = I_{bgd}(\mathbf{x}_i) + \sum_{hkl} |q_{hkl}|^2
               \chi_n(\mathbf{u}(\mathbf{x}_i)) f^2_{hkl}(\mathbf{x}_i),

            where :math:`q_{hkl}` are the structure factors and :math:`\chi(\mathbf{u}(\mathbf{x}))`
            are the projection maps of the sample, and :math:`f_{hkl}(\mathbf{x})` are the standard
            reflection profiles.

            The Poisson negative log-likelihood criterion is given by:

            .. math::
                \varepsilon^{NLL} = \sum_{ni} \varepsilon_n^{NLL}(\mathbf{x}_i) = 
                \sum_{ni} \log \mathrm{P}(I_n(\mathbf{x}_i), \hat{I}_n(\mathbf{x}_i)),

            where the likelihood :math:`\mathrm{P}` follows the Poisson distribution
            :math:`\log \mathrm{P}(I, \lambda) = I \log \lambda - I`.

            The least-squares criterion is given by:

            .. math::
                \varepsilon^{LS} = \sum_{ni} \varepsilon_n^{LS}(\mathbf{x}_i) =
                \sum_{ni} f\left( \frac{I_n(\mathbf{x}_i) - \hat{I}_n(\mathbf{x}_i)}{\sigma_I} \right),

            where :math:`f(x)` is either l2, l1, or Huber loss function, and :math:`\sigma_I` is the
            standard deviation of measured photon counts for a given diffraction streak.

        Raises:
            ValueError : If the `loss` argument is invalid.

        Returns:
            The criterion and the jacobian matrix.
        """
        if not fit_intercept:
            x[:self.idxs[-1] + 1] = 0.0

        if kind == 'poisson':
            crit, jac = poisson_criterion(x, ij=self.ij, shape=self.parent().shape[1:], I0=self.I_raw,
                                          bgd=self.bgd, xtal_bi=self.xtal_bi, prof=self.prof,
                                          fidxs=self.fidxs, idxs=self.idxs, hkl_idxs=self.hkl_idxs,
                                          num_threads=self.num_threads)
        elif kind == 'least_squares':
            crit, jac = ls_criterion(x, ij=self.ij, shape=self.parent().shape[1:], I0=self.I_raw,
                                     bgd=self.bgd, xtal_bi=self.xtal_bi, prof=self.prof, loss=loss,
                                     fidxs=self.fidxs, idxs=self.idxs, hkl_idxs=self.hkl_idxs,
                                     num_threads=self.num_threads)
        else:
            raise ValueError(f"'kind' keyword is invalid: {kind}")

        if not fit_intercept:
            jac[:self.idxs[-1] + 1] = 0.0

        return crit, jac

    def gain(self, x: np.ndarray, kind: str='poisson', loss: str='l2') -> np.ndarray:
        """Return the fitness gain for every diffraction order.

        Args:
            x : Current estimate of crystal structure factors and intercept values.
            kind : Choose between the negative log likelihood Poisson criterion ('poisson')
                and the least squares criterion ('least_squares')
            loss : Loss function to use in the least squares criterion. The following
                keyword arguments are allowed:

                * `l1`: L1 loss (absolute) function.
                * `l2` : L2 loss (squared) function.
                * `Huber` : Huber loss function.

        Returns:
            Array of fitness gain values for each diffraction order.
        """
        x0 = self.init_estimate()

        if kind == 'poisson':
            crit0 = poisson_criterion(x0, shape=self.parent().shape[1:], I0=self.I_raw, ij=self.ij,
                                      bgd=self.bgd, xtal_bi=self.xtal_bi, prof=self.prof,
                                      fidxs=self.fidxs, idxs=self.idxs, hkl_idxs=self.hkl_idxs,
                                      oidxs=self.hkl_idxs, num_threads=self.num_threads)[0]
            crit = poisson_criterion(x, shape=self.parent().shape[1:], I0=self.I_raw, ij=self.ij,
                                     bgd=self.bgd, xtal_bi=self.xtal_bi, prof=self.prof,
                                     fidxs=self.fidxs, idxs=self.idxs, hkl_idxs=self.hkl_idxs,
                                     oidxs=self.hkl_idxs, num_threads=self.num_threads)[0]
        elif kind == 'least_squares':
            crit0 = ls_criterion(x0, shape=self.parent().shape[1:], prof=self.prof, I0=self.I_raw,
                                 ij=self.ij, bgd=self.bgd, xtal_bi=self.xtal_bi, idxs=self.idxs,
                                 fidxs=self.fidxs, hkl_idxs=self.hkl_idxs, oidxs=self.hkl_idxs,
                                 loss=loss, num_threads=self.num_threads)[0]
            crit = ls_criterion(x, shape=self.parent().shape[1:], prof=self.prof, I0=self.I_raw,
                                ij=self.ij, bgd=self.bgd, xtal_bi=self.xtal_bi, idxs=self.idxs,
                                fidxs=self.fidxs, hkl_idxs=self.hkl_idxs, oidxs=self.hkl_idxs,
                                loss=loss, num_threads=self.num_threads)[0]
        else:
            raise ValueError(f"'kind' keyword is invalid: {kind}")

        return crit0 - crit

    def merge_hkl(self, symmetry: Optional[str]=None) -> IntensityScaler:
        """Merge symmetrical reflection during the iterative update. Given the supplied symmetry,
        the structure factors for the symmetric reflection are assumed to be identical during
        the scaling procedure.

        Args:
            symmetry : Crystallographic point group symmetry. The following keyword  arguments are
                allowed:

                * `mmm`: Centrosymmetric othorhombic point group.
                * `4mm` : Tetragonal point group.
                * `None` : No symmetry.

        Returns:
            A new :class:`IntensityScaler` object with the updated symmetry.
        """
        obj = self.unmerge_hkl()
        if symmetry == 'mmm':
            hkl_sym = np.abs(obj.hkl)
        elif symmetry == '4mm':
            hkl_sym = np.abs(obj.hkl)
            hkl_sym = np.concatenate((np.sort(hkl_sym[:, :2], axis=1), hkl_sym[:, 2:]), axis=1)
        elif symmetry is None:
            hkl_sym = obj.hkl
        else:
            raise ValueError('Invalid symmetry keyword')

        hkl, hkl_idxs = np.unique(hkl_sym, return_inverse=True, axis=0)
        return self.replace(hkl=hkl, hkl_idxs=np.asarray(hkl_idxs, dtype=np.uint32))

    def merge_sfac(self, x: np.ndarray, symmetry: Optional[str]=None) -> Dict[str, np.ndarray]:
        """Return a merged list of structure factors.

        Args:
            x : Current estimate of crystal structure factors and intercept values.
            symmetry : Symmetry of the crystal. Can be one of the following:

                * `4mm` : (h, k, l) = (-h, -k, -l) = (k, h, l).
                * `mmm` : (h, k, l) = (-h, -k, -l).
                * `None` : No symmetry.

        Returns:
            A dictionary of the following attributes:

            * `hkl` : Merged list of Miller indices.
            * `sfac` : Merged list of structure factors.
            * `serr` : List of structure factor uncertainties.
            * `cnts` : Number of measurements of the given reflection.
        """
        obj = self.merge_hkl(symmetry=symmetry)

        sfac = np.zeros(obj.hkl_idxs.max() + 1, dtype=float)
        norm = np.zeros(obj.hkl_idxs.max() + 1, dtype=int)
        cnts = np.zeros(obj.hkl_idxs.max() + 1, dtype=int)
        np.add.at(norm, obj.hkl_idxs, self.iidxs[1:] - self.iidxs[:-1])
        np.add.at(sfac, obj.hkl_idxs, np.exp(x[self.idxs[-1] + self.hkl_idxs + 1]))
        np.add.at(sfac, obj.hkl_idxs, np.ones(self.iidxs.size - 1, dtype=int))
        sfac = np.where(cnts, sfac / cnts, 0.0)
        sfac_err = np.where(norm, np.sqrt(sfac / norm), 0.0)

        sfac /= np.mean(sfac_err)
        sfac_err /= np.mean(sfac_err)
        return {'hkl': obj.hkl, 'sfac': sfac, 'serr': sfac_err, 'cnts': cnts}

    def unmerge_hkl(self) -> IntensityScaler:
        """Unmerge symmetrical reflection during the iterative update. Each reflection will be
        updated separately during the scaling procedure.

        Returns:
            A new :class:`IntensityScaler` object with no merging of reflections.
        """
        return self.replace(hkl=None, hkl_idxs=None)

    def model(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Unmerge the structure factors and intercepts.

        Args:
            x : Current estimate of crystal structure factors and intercept values.

        Returns:
            An array of unmerged structure factors and intercepts.
        """
        sfac = np.exp(x[self.idxs[-1] + self.hkl_idxs[self.idxs] + 1])
        return x[self.idxs], sfac

    def update_xtal(self, x: np.ndarray, bandwidth: float) -> Tuple[IntensityScaler, np.ndarray]:
        """Generate a new crystal map using the kernel regression.

        Args:
            x : Current estimate of crystal structure factors and intercept values.
            bandwidth : kernel bandwidth in pixels.

        Returns:
            New :class:`IntensityScaler` container with the updated crystal efficiency map.
        """
        sfac = np.sqrt(np.exp(x[self.idxs[-1] + self.hkl_idxs[self.idxs] + 1]), dtype=np.float32)
        I_hat = unmerge_signal(x, ij=self.ij, shape=self.parent().shape[1:], I0=self.I_raw,
                               bgd=self.bgd, xtal_bi=self.xtal_bi, prof=self.prof,
                               fidxs=self.fidxs, idxs=self.idxs, hkl_idxs=self.hkl_idxs,
                               num_threads=self.num_threads)
        pupils = np.zeros(self.xtal.shape, dtype=np.float32)
        for index, (f0, f1) in enumerate(zip(self.fidxs, self.fidxs[1:])):
            pupil, roi = kr_grid(y=I_hat[f0:f1] / sfac[f0:f1],  grid=(self.xtal.x, self.xtal.y),
                                 x=self.xmap[f0:f1, :2], w=self.prof[f0:f1] * sfac[f0:f1],
                                 sigma=bandwidth, num_threads=self.num_threads)
            pupils[index, roi[0]:roi[1], roi[2]:roi[3]] = pupil

        norm = np.mean(pupils)
        x[self.iidxs.size - 1:] += np.log(norm)
        return self.replace(xtal=self.xtal.replace(val=pupils / norm), xtal_bi=None), x

    def update_table(self, x: np.ndarray):
        """Update the parent :class:`cbclib.CBCTable` table.

        Args:
            x : Current estimate of crystal structure factors and intercept values.
        """
        self.parent().table['sfac'] = self.model(x)[1]
        self.parent().table['xtal'] = self.xtal_bi

    def train(self, bandwidth: float, n_iter: int=10, f_tol: float=0.0, fit_intercept: bool=True,
              kind: str='poisson', loss: str='l2', max_iter: int=10, x0: Optional[np.ndarray]=None,
              verbose: bool=True) -> Tuple[IntensityScaler, np.ndarray]:
        """Perform an iterative update of sample projection maps (xtal) and crystal structure
        factors (sfac) until the error between the predicted intensity Bragg profiles and
        experimental patterns converges to a minimum. The method provides two minimisation
        criteria: Poisson likelihood and least squares error.

        Args:
            bandwidth : Kernel bandwidth used in the diffraction efficiency map (xtal) update.
            n_iter : Maximum number of iterations.
            f_tol : Tolerance for termination by the change of the average error. The iteration
                stops when :math:`(f^k - f^{k + 1}) / max(|f^k|, |f^{k + 1}|) <= f_{tol}`.
            fit_intercept : Update intercepts if True.
            kind : Choose between the negative log likelihood Poisson criterion ('poisson')
                and the least squares criterion ('least_squares')
            loss : Loss function to use in the least squares criterion. The following
                keyword arguments are allowed:

                * `l1`: L1 loss (absolute) function.
                * `l2` : L2 loss (squared) function.
                * `Huber` : Huber loss function.

            max_iter : Maximum number of iterations.
            x0 : Preliminary estimate of crystal structure factors and intercepts.
            verbose : Set verbosity of the computation process.

        Returns:
            An updated intensity scaler object and a new estimate of crystal structure factors
            and intercepts.
        """
        x = self.init_estimate() if x0 is None else x0

        itor = tqdm(range(1, n_iter + 1), disable=not verbose,
                    bar_format='{desc} {percentage:3.0f}% {bar} '\
                    'Iteration {n_fmt} / {total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')

        obj = self.replace()
        errors = [obj.fitness(x, fit_intercept=fit_intercept, kind=kind, loss=loss)[0]]
        if verbose:
            itor.set_description(f"Error = {float(errors[-1]):.6f}")

        for _ in itor:
            res = minimize(obj.fitness, x, args=(fit_intercept, kind, loss), jac=True,
                           method='l-bfgs-b', options={'maxiter': max_iter})

            new_obj, x = obj.update_xtal(res.x, bandwidth=bandwidth)
            errors.append(new_obj.fitness(x, fit_intercept=fit_intercept, kind=kind, loss=loss)[0])

            if verbose:
                itor.set_description(f"Error = {float(errors[-1]):.6f}")

            if (errors[-2] - errors[-1]) / max(abs(errors[-2]), abs(errors[-1])) > f_tol:
                obj = new_obj
            else:
                break

        return obj, res.x

    def export_sfac(self, x: np.ndarray, path: str, symmetry: Optional[str]=None):
        """Export structure factors to a text hkl file.

        Args:
            x : Current estimate of crystal structure factors and intercept values.
            path : Path to the output file.
            symmetry : Symmetry of the crystal. Can be one of the following:

                * `4mm` : (h, k, l) = (-h, -k, -l) = (k, h, l).
                * `mmm` : (h, k, l) = (-h, -k, -l).
                * `None` : No symmetry.
        """
        sf_data = self.merge_sfac(x, symmetry)
        sf_arr = np.concatenate((sf_data['hkl'], sf_data['sfac'][:, None],
                                 sf_data['serr'][:, None], sf_data['cnts'][:, None]), axis=1)
        np.savetxt(path, sf_arr, ' %4d %4d %4d %10.2f %10.2f %4d')
