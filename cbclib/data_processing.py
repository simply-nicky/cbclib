""":class:`cbclib.CrystData` stores all the data necessarry to process measured convergent
beam crystallography patterns and provides a suite of data processing tools to wor with the
detector data.

Examples:
    Load all the necessary data using a :func:`cbclib.CrystData.load` function.

    >>> import cbclib as cbc
    >>> inp_file = cbc.CXIStore('data.cxi')
    >>> data = cbc.CrystData(inp_file)
    >>> data = data.load()
"""
from __future__ import annotations
from multiprocessing import cpu_count
from typing import Dict, List, Optional, TypeVar, Union, cast
from dataclasses import dataclass, field
from weakref import ref
import numpy as np
import pandas as pd
from .cbc_setup import Basis, ScanSamples, ScanSetup, Streaks, CBDModel
from .cxi_protocol import CrystProtocol, FileStore, Kinds
from .data_container import StringFormatter, DataContainer, Transform, ReferenceType
from .streak_finder import PatternsStreakFinder
from .annotations import (Indices, IntSequence, NDArray, NDArrayLike, NDBoolArray, NDIntArray,
                          NDRealArray, ROIType, Shape)
from .src import binterpolate, kr_grid, label, median, robust_mean, robust_lsq, Regions, Structure

def read_hdf(input_file: FileStore, attributes: Union[str, List[str]],
             idxs: Optional[IntSequence]=None, transform: Optional[Transform] = None,
             processes: int=1, verbose: bool=True
             ) -> Union[CrystData, CrystDataPart, CrystDataFull]:
    """Load data attributes from the input files in `files` file handler object.

    Args:
        attributes : List of attributes to load. Loads all the data attributes contained in
            the file(s) by default.
        idxs : List of frame indices to load.
        processes : Number of parallel workers used during the loading.
        verbose : Set the verbosity of the loading process.

    Raises:
        ValueError : If attribute is not existing in the input file(s).
        ValueError : If attribute is invalid.

    Returns:
        New :class:`CrystData` object with the attributes loaded.
    """
    if input_file.is_empty():
        input_file.update()
    shape = input_file.read_frame_shape()

    if idxs is None:
        idxs = np.arange(input_file.size)
    idxs = np.atleast_1d(idxs)

    data_dict: Dict[str, NDArray] = {'frames': idxs}

    for attr in StringFormatter.str_to_list(attributes):
        if attr not in input_file.attributes():
            raise ValueError(f"No '{attr}' attribute in the input files")

        if transform and shape[0] * shape[1]:
            ss_idxs, fs_idxs = np.indices(shape)
            ss_idxs, fs_idxs = transform.index_array(ss_idxs, fs_idxs)
            data = input_file.load(attr, idxs=idxs, ss_idxs=ss_idxs,
                                   fs_idxs=fs_idxs, processes=processes,
                                   verbose=verbose)
        else:
            data = input_file.load(attr, idxs=idxs, processes=processes,
                                   verbose=verbose)

        data_dict[attr] = data

    if 'data' in data_dict:
        if 'whitefield' in data_dict:
            return CrystDataFull(**data_dict)
        return CrystDataPart(**data_dict)
    return CrystData(**data_dict)

def write_hdf(container: Union[CrystData, CrystDataPart, CrystDataFull], output_file: FileStore,
              attributes: Union[str, List[str], None]=None, good_frames: Optional[Indices]=None,
              transform: Optional[Transform] = None, input_file: Optional[FileStore]=None,
              mode: str='append', idxs: Optional[Indices]=None):
    """Save data arrays of the data attributes contained in the container to an output file.

    Args:
        attributes : List of attributes to save. Saves all the data attributes contained in
            the container by default.
        apply_transform : Apply `transform` to the data arrays if True.
        mode : Writing modes. The following keyword values are allowed:

            * `append` : Append the data array to already existing dataset.
            * `insert` : Insert the data under the given indices `idxs`.
            * `overwrite` : Overwrite the existing dataset.

        idxs : Indices where the data is saved. Used only if ``mode`` is set to 'insert'.

    Raises:
        ValueError : If the ``output_file`` is not defined inside the container.
    """
    if transform:
        if input_file is None:
            raise ValueError("'input_file' is not defined inside the container")

        shape = input_file.read_frame_shape()

    if attributes is None:
        attributes = list(container.contents())

    if good_frames is None:
        good_frames = np.arange(container.shape[0])

    for attr in StringFormatter.str_to_list(attributes):
        data = np.asarray(container.get(attr))
        if data is not None:
            kind = container.get_kind(attr)

            if kind in (Kinds.STACK, Kinds.SEQUENCE):
                data = data[good_frames]

            if transform:
                if kind in (Kinds.STACK, Kinds.FRAME):
                    out = np.zeros(data.shape[:-2] + shape, dtype=data.dtype)
                    data = transform.backward(data, out)

            output_file.save(attr, data, mode=mode, idxs=idxs)

@dataclass
class CrystDataBase(CrystProtocol, DataContainer):
    """Convergent beam crystallography data container class. Takes a :class:`cbclib.CXIStore` file
    handler. Provides an interface to work with the detector images and detect the diffraction
    streaks. Also provides an interface to load from a file and save to a file any of the data
    attributes. The data frames can be tranformed using any of the :class:`cbclib.Transform`
    classes.

    Args:
        input_file : Input file :class:`cbclib.CXIStore` file handler.
        transform : An image transform object.
        num_threads : Number of threads used in the calculations.
        output_file : On output file :class:`cbclib.CXIStore` file handler.
        data : Detector raw data.
        mask : Bad pixels mask.
        frames : List of frame indices inside the container.
        whitefield : Measured frames' white-field.
        snr : Signal-to-noise ratio.
        whitefields : A set of white-fields generated for each pattern separately.
    """
    frames      : NDIntArray

    data        : Optional[NDRealArray]
    whitefield  : Optional[NDRealArray]

    mask        : Optional[NDBoolArray] = field(default=None)
    std         : Optional[NDRealArray] = field(default=None)
    snr         : Optional[NDRealArray] = field(default=None)
    scales      : Optional[NDRealArray] = field(default=None)
    num_threads : int = field(default=cpu_count())

    @property
    def shape(self) -> Shape:
        shape = [0, 0, 0]
        for attr, data in self.items():
            if data is not None and self.get_kind(attr) == Kinds.SEQUENCE:
                shape[0] = data.shape[0]
                break

        for attr, data in self.items():
            if data is not None and self.get_kind(attr) == Kinds.FRAME:
                shape[1:] = data.shape
                break

        for attr, data in self.items():
            if data is not None and self.get_kind(attr) == Kinds.STACK:
                shape[:] = data.shape
                break

        return tuple(shape)

    def select_frames(self, idxs: Optional[Indices]=None):
        """Return a new :class:`CrystData` object with the new mask.

        Args:
            mask : New mask array.
            update : Multiply the new mask and the old one if 'multiply', use the
                new one if 'reset'.

        Raises:
            ValueError : If the mask shape is incompatible with the data.
            ValueError : If there is no ``data`` inside the container.

        Returns:
            New :class:`CrystData` object with the updated ``mask``.
        """
        data_dict = {}
        for attr in self.contents():
            if self.get_kind(attr) in (Kinds.SEQUENCE, Kinds.STACK):
                data_dict[attr] = self.get(attr)[idxs]
            else:
                data_dict[attr] = self.get(attr)[idxs]
        return self.replace(**data_dict)

Cryst = TypeVar("Cryst", bound="CrystDataPartBase")

@dataclass
class CrystDataPartBase(CrystDataBase):
    data        : NDRealArray
    mask        : NDBoolArray = field(default_factory=lambda: np.array([], dtype=bool))

    def __post_init__(self):
        if self.mask.shape != self.shape[1:]:
            self.mask = np.ones(self.shape[1:], dtype=bool)

    def mask_region(self: Cryst, roi: ROIType) -> Cryst:
        """Return a new :class:`CrystData` object with the updated mask. The region
        defined by the `[y_min, y_max, x_min, x_max]` will be masked out.

        Args:
            roi : Bad region of interest in the detector plane. A set of four
                coordinates `[y_min, y_max, x_min, x_max]`.

        Raises:
            ValueError : If there is no ``data`` inside the container.

        Returns:
            New :class:`CrystData` object with the updated ``mask``.
        """
        mask = np.copy(self.mask)
        mask[roi[0]:roi[1], roi[2]:roi[3]] = False
        return self.replace(mask=mask)

    def import_mask(self: Cryst, mask: NDBoolArray, update: str='reset') -> Cryst:
        """Return a new :class:`CrystData` object with the new mask.

        Args:
            mask : New mask array.
            update : Multiply the new mask and the old one if 'multiply', use the
                new one if 'reset'.

        Raises:
            ValueError : If the mask shape is incompatible with the data.
            ValueError : If there is no ``data`` inside the container.

        Returns:
            New :class:`CrystData` object with the updated ``mask``.
        """
        if mask.shape != self.shape[1:]:
            raise ValueError('mask and data have incompatible shapes: '\
                             f'{mask.shape:s} != {self.shape[1:]:s}')

        if update == 'reset':
            return self.replace(mask=mask)
        if update == 'multiply':
            return self.replace(mask=mask * self.mask)

        raise ValueError(f'Invalid update keyword: {update:s}')

    def reset_mask(self: Cryst) -> Cryst:
        """Reset bad pixel mask. Every pixel is assumed to be good by default.

        Raises:
            ValueError : If there is no ``data`` inside the container.

        Returns:
            New :class:`CrystData` object with the default ``mask``.
        """
        return self.replace(mask=np.array([], dtype=bool))

    def update_mask(self: Cryst, method: str='no-bad', vmin: int=0, vmax: int=65535,
                    snr_max: float=3.0, roi: Optional[ROIType]=None) -> Cryst:
        """Return a new :class:`CrystData` object with the updated bad pixels mask.

        Args:
            method : Bad pixels masking methods. The following keyword values are
                allowed:

                * 'all-bad' : Mask out all pixels.
                * 'no-bad' (default) : No bad pixels.
                * 'range' : Mask the pixels which values lie outside of (`vmin`,
                  `vmax`) range.
                * 'snr' : Mask the pixels which SNR values lie exceed the SNR
                  threshold `snr_max`. The snr is given by
                  :code:`abs(data - whitefield) / sqrt(whitefield)`.

            vmin : Lower intensity bound of 'range-bad' masking method.
            vmax : Upper intensity bound of 'range-bad' masking method.
            snr_max : SNR threshold.
            roi : Region of the frame undertaking the update. The whole frame is updated
                by default.

        Raises:
            ValueError : If there is no ``data`` inside the container.
            ValueError : If there is no ``snr`` inside the container.
            ValueError : If ``method`` keyword is invalid.
            ValueError : If ``vmin`` is larger than ``vmax``.

        Returns:
            New :class:`CrystData` object with the updated ``mask``.
        """
        if vmin >= vmax:
            raise ValueError('vmin must be less than vmax')
        if roi is None:
            roi = (0, self.shape[1], 0, self.shape[2])

        data = (self.data * self.mask)[:, roi[0]:roi[1], roi[2]:roi[3]]

        if method == 'all-bad':
            mask = np.zeros(self.shape[1:], dtype=bool)
        elif method == 'no-bad':
            mask = np.ones(self.shape[1:], dtype=bool)
        elif method == 'range':
            mask = np.all((data >= vmin) & (data < vmax), axis=0)
        elif method == 'snr':
            if self.snr is None:
                raise ValueError('No snr in the container')

            snr = self.snr[:, roi[0]:roi[1], roi[2]:roi[3]]
            mask = np.mean(np.abs(snr), axis=0) < snr_max
        else:
            raise ValueError(f'Invalid method argument: {method:s}')

        new_mask = np.copy(self.mask)
        new_mask[roi[0]:roi[1], roi[2]:roi[3]] &= mask
        return self.replace(mask=new_mask)

    def update_whitefield(self, method: str='median', frames: Optional[Indices]=None,
                          r0: float=0.0, r1: float=0.5, n_iter: int=12,
                          lm: float=9.0) -> CrystDataFull:
        """Return a new :class:`CrystData` object with new whitefield.

        Args:
            method : Choose method for white-field generation. The following keyword
                values are allowed:

                * 'median' : Taking a median through the stack of frames.
                * 'robust-mean' : Finding a robust mean through the stack of frames.

            frames : List of frames to use for the white-field estimation.
            r0 : A lower bound guess of ratio of inliers. We'd like to make a sample
                out of worst inliers from data points that are between `r0` and `r1`
                of sorted residuals.
            r1 : An upper bound guess of ratio of inliers. Choose the `r0` to be as
                high as you are sure the ratio of data is inlier.
            n_iter : Number of iterations of fitting a gaussian with the FLkOS
                algorithm.
            lm : How far (normalized by STD of the Gaussian) from the mean of the
                Gaussian, data is considered inlier.

        Raises:
            ValueError : If there is no ``data`` inside the container.
            ValueError : If ``method`` keyword is invalid.

        Returns:
            New :class:`CrystData` object with the updated ``whitefield``.
        """
        if frames is None:
            frames = np.arange(self.shape[0])
        std = self.std

        if method == 'median':
            whitefield = median(inp=self.data[frames] * self.mask, axis=0,
                                num_threads=self.num_threads)
        elif method == 'robust-mean':
            whitefield = robust_mean(inp=self.data[frames] * self.mask, axis=0, r0=r0,
                                     r1=r1, n_iter=n_iter, lm=lm,
                                     num_threads=self.num_threads)
        elif method == 'robust-mean-scale':
            whitefield, std = robust_mean(inp=self.data[frames] * self.mask, axis=0,
                                          r0=r0, r1=r1, n_iter=n_iter, lm=lm,
                                          return_std=True, num_threads=self.num_threads)
        else:
            raise ValueError('Invalid method argument')

        return CrystDataFull(**dict(self.to_dict(), whitefield=whitefield, std=std))

@dataclass
class CrystData(CrystDataBase):
    data        : Optional[NDRealArray] = field(default=None)
    whitefield  : Optional[NDRealArray] = field(default=None)

@dataclass
class CrystDataPart(CrystDataPartBase):
    data        : NDRealArray
    whitefield  : Optional[NDRealArray] = field(default=None)

@dataclass
class CrystDataFull(CrystDataPartBase):
    whitefield  : NDRealArray
    scales      : NDRealArray = field(default_factory=lambda: np.array([]))

    def __post_init__(self):
        if self.scales.shape != (self.frames.size,):
            self.scales = np.ones(self.frames.size)

    def streak_detector(self, structure: Structure) -> StreakDetector:
        """Return a new :class:`cbclib.StreakDetector` object that detects lines in SNR frames.

        Raises:
            ValueError : If there is no ``whitefield`` inside the container.
            ValueError : If there is no ``snr`` inside the container.

        Returns:
            A CBC pattern detector based on :class:`cbclib.bin.LSD` Line Segment Detection [LSD]_
            algorithm.
        """
        if self.snr is None:
            raise ValueError('No snr in the container')

        parent = cast(ReferenceType[CrystDataFull], ref(self))
        return StreakDetector(data=self.snr, mask=self.mask, parent=parent, frames=self.frames,
                              structure=structure, num_threads=self.num_threads)

    def region_detector(self, structure: Structure):
        if self.snr is None:
            raise ValueError('No snr in the container')

        parent = cast(ReferenceType[CrystDataFull], ref(self))
        return RegionDetector(data=self.snr, mask=self.mask, parent=parent, frames=self.frames,
                              structure=structure, num_threads=self.num_threads)

    def model_detector(self, basis: Basis, samples: ScanSamples, setup: ScanSetup) -> ModelDetector:
        """Return a new :class:`cbclib.ModelDetector` object that finds the diffracted streaks
        in SNR frames based on the solution of sample and indexing refinement.

        Args:
            basis : Indexing solution.
            samples : Sample refinement solution.
            setup : Experimental setup.

        Raises:
            ValueError : If there is no ``whitefield`` inside the container.
            ValueError : If there is no ``snr`` inside the container.

        Returns:
            A CBC pattern detector based on :class:`cbclib.CBDModel` CBD pattern prediction model.
        """
        if self.snr is None:
            raise ValueError('No snr in the container')

        parent = cast(ReferenceType[CrystDataFull], ref(self))
        return ModelDetector(data=self.snr, parent=parent, frames=self.frames,
                             model=CBDModel(basis, samples, setup, num_threads=self.num_threads))

    def update_std(self, method="robust-scale", frames: Optional[Indices]=None,
                   r0: float=0.0, r1: float=0.5, n_iter: int=12, lm: float=9.0) -> CrystDataFull:
        if frames is None:
            frames = np.arange(self.shape[0])

        if method == "robust-scale":
            _, std = robust_mean(inp=self.data[frames] * self.mask, axis=0, r0=r0, r1=r1,
                                 n_iter=n_iter, lm=lm, return_std=True,
                                 num_threads=self.num_threads)
        elif method == "poisson":
            std = np.sqrt(self.whitefield)
        else:
            raise ValueError(f"Invalid method argument: {method}")

        return self.replace(std=std)

    def update_snr(self) -> CrystDataFull:
        """Return a new :class:`CrystData` object with new background corrected detector
        images.

        Raises:
            ValueError : If there is no ``whitefield`` inside the container.

        Returns:
            New :class:`CrystData` object with the updated ``cor_data``.
        """
        if self.std is None:
            raise ValueError("No snr in the container")

        whitefields = self.scales[:, None, None] * self.whitefield
        snr = np.where(self.std, (self.data * self.mask - whitefields) / self.std, 0.0)
        return self.replace(snr=snr)

    def scale_whitefield(self, method: str="robust-lsq", r0: float=0.0, r1: float=0.5,
                         n_iter: int=12, lm: float=9.0) -> CrystDataFull:
        """Return a new :class:`CrystData` object with a new set of whitefields. A set of
        backgrounds is generated by robustly fitting a design matrix `W` to the measured
        patterns.

        Args:
            method : Choose one of the following methods to scale the white-field:

                * "median" : By taking a median of data and whitefield.
                * "robust-lsq" : By solving a least-squares problem with truncated
                  with the fast least k-th order statistics (FLkOS) estimator.

            r0 : A lower bound guess of ratio of inliers. We'd like to make a sample
                out of worst inliers from data points that are between `r0` and `r1`
                of sorted residuals.
            r1 : An upper bound guess of ratio of inliers. Choose the `r0` to be as
                high as you are sure the ratio of data is inlier.
            n_iter : Number of iterations of fitting a gaussian with the FLkOS
                algorithm.
            lm : How far (normalized by STD of the Gaussian) from the mean of the
                Gaussian, data is considered inlier.

        Raises:
            ValueError : If there is no ``data`` inside the container.
            ValueError : If there is no ``whitefield`` inside the container.

        Returns:
            An array of scale factors for each frame in the container.
        """
        if self.std is None:
            raise ValueError("No snr in the container")

        mask = self.mask & (self.std > 0.0)
        y: NDRealArray = np.where(mask, self.data / self.std, 0.0)[:, mask]
        W: NDRealArray = np.where(mask, self.whitefield / self.std, 0.0)[None, mask]

        if method == "robust-lsq":
            scales = robust_lsq(W=W, y=y, axis=1, r0=r0, r1=r1, n_iter=n_iter, lm=lm,
                                num_threads=self.num_threads)
            return self.replace(scales=scales.ravel())

        if method == "median":
            scales = median(y * W, axis=1, num_threads=self.num_threads)[:, None] / \
                     median(W * W, axis=1, num_threads=self.num_threads)[:, None]
            return self.replace(scales=scales.ravel())

        raise ValueError(f"Invalid method argument: {method}")

DetBase = TypeVar("DetBase", bound="DetectorBase")

@dataclass
class DetectorBase(DataContainer):
    frames          : NDIntArray
    data            : NDRealArray
    num_threads     : int
    parent          : ReferenceType[CrystDataFull]

    @property
    def shape(self) -> Shape:
        return self.data.shape

    def get_frames(self: DetBase, idxs: Indices) -> DetBase:
        raise NotImplementedError

    def clip(self: DetBase, vmin: NDArrayLike, vmax: NDArrayLike) -> DetBase:
        return self.replace(data=np.clip(self.data, vmin, vmax))

    def export_coordinates(self, frames: NDIntArray, y: NDIntArray, x: NDIntArray) -> pd.DataFrame:
        table = {'bgd': self.parent().scales[frames] * self.parent().whitefield[y, x],
                 'frames': self.frames[frames], 'I_raw': self.parent().data[frames, y, x],
                 'snr': self.data[frames, y, x], 'x': x, 'y': y}
        return pd.DataFrame(table)

MDet = TypeVar("MDet", bound="MaskedDetector")

@dataclass
class MaskedDetector(DetectorBase):
    mask            : NDBoolArray

    def downscale(self: MDet, ratio: float, sigma: float) -> MDet:
        x, y = np.arange(0, self.shape[-1]), np.arange(0, self.shape[-2])
        pts = np.stack(np.meshgrid(x, y), axis=-1)

        x_scaled = ratio * np.arange(0, self.shape[-1] / ratio)
        y_scaled = ratio * np.arange(0, self.shape[-2] / ratio)
        pts_scaled = np.stack(np.meshgrid(x_scaled, y_scaled), axis=-1)

        data_scaled = kr_grid(self.data, pts, (x_scaled, y_scaled), sigma=sigma,
                              num_threads=self.num_threads)[0]
        mask_scaled = binterpolate(np.asarray(self.mask, dtype=float), (x, y), pts_scaled)
        return self.replace(data=data_scaled, mask=np.asarray(mask_scaled, dtype=bool))

    def get_frames(self: MDet, idxs: Indices) -> MDet:
        return self.replace(data=self.data[idxs], mask=self.mask[idxs], frames=self.frames[idxs])

class StreakDetectorBase(DetectorBase):
    def export_table(self, streaks: Streaks, width: float,
                     kernel: str='rectangular') -> pd.DataFrame:
        """Export normalised pattern into a :class:`pandas.DataFrame` table.

        Args:
            streaks : A set of diffraction streaks.
            width : Width of diffraction streaks in pixels.
            kernel : Choose one of the supported kernel functions [Krn]_. The following
                kernels are available:

                * 'biweigth' : Quartic (biweight) kernel.
                * 'gaussian' : Gaussian kernel.
                * 'parabolic' : Epanechnikov (parabolic) kernel.
                * 'rectangular' : Uniform (rectangular) kernel.
                * 'triangular' : Triangular kernel.

        Raises:
            ValueError : If there is no ``streaks`` inside the container.

        Returns:
            List of :class:`pandas.DataFrame` tables for each frame in ``frames`` if
            ``concatenate`` is False, a single :class:`pandas.DataFrame` otherwise. Table
            contains the following information:

            * `frames` : Frame index.
            * `x`, `y` : Pixel coordinates.
            * `snr` : Signal-to-noise values.
            * `rp` : Reflection profiles.
            * `I_raw` : Measured intensity.
            * `bgd` : Background values.
        """
        if self.parent() is None:
            raise ValueError('Invalid parent: the parent data container was deleted')

        table = streaks.pattern_dataframe(width, shape=self.shape, kernel=kernel)
        table2 = self.export_coordinates(table['frames'].to_numpy(),
                                         table['y'].to_numpy(), table['x'].to_numpy())
        return table.merge(table2, how='left', on=['frames', 'y', 'x'])

@dataclass
class StreakDetector(StreakDetectorBase, MaskedDetector, PatternsStreakFinder):
    pass

@dataclass
class RegionDetector(MaskedDetector):
    structure   : Structure

    def detect_regions(self, vmin: float, npts: int) -> List[Regions]:
        return label((self.data > vmin) & self.mask, self.structure, npts,
                     num_threads=self.num_threads)

    def export_table(self, regions: List[Regions]) -> pd.DataFrame:
        frames, y, x = [], [], []
        for frame, pattern in zip(self.frames, regions):
            size = sum(region.size for region in pattern)
            frames.extend(size * [frame,])
            y.extend(pattern.y)
            x.extend(pattern.x)
        return self.export_coordinates(np.array(frames), np.array(y), np.array(x))

@dataclass
class ModelDetector(StreakDetectorBase, DetectorBase):
    """A streak detector class based on the CBD pattern prediction. Uses :class:`cbclib.CBDModel` to
    predict a pattern and filters out all the predicted streaks, that correspond to the measured
    intensities above the certain threshold. Provides an interface to generate an indexing tabular
    data.

    Args:
        snr : Signal-to-noise ratio patterns.
        frames : Frame indices of the detector images.
        parent : A reference to the parent :class:`cbclib.CrystData` container.
        model : A convergent beam diffraction model.
        streaks : A dictionary of detected :class:`cbclib.Streaks` streaks.
    """
    model   : CBDModel

    def get_frames(self, idxs: Indices) -> ModelDetector:
        return self.replace(data=self.data[idxs], frames=self.frames[idxs], model=self.model[idxs])

    def count_snr(self, streaks: Streaks, hkl: NDIntArray, width: float,
                  kernel: str='rectangular') -> NDRealArray:
        r"""Count the average signal-to-noise ratio for a set of reciprocal lattice points `hkl`.

        Args:
            hkl : Miller indices of reciprocal lattice points.
            width : Diffraction streak width in pixels.

        Returns:
            An array of average SNR values for each reciprocal lattice point in `hkl`.
        """
        snr = np.zeros(hkl.shape[0])
        cnts = np.zeros(hkl.shape[0], dtype=int)
        df = streaks.pattern_dataframe(width=width, shape=self.shape[1:], kernel=kernel)
        np.add.at(snr, df['hkl_id'], self.data[df['frames'], df['y'], df['x']])
        np.add.at(cnts, df['hkl_id'], np.ones(df.shape[0], dtype=int))
        return np.where(cnts, snr / cnts, 0.0)

    def detect(self, hkl: NDIntArray, hkl_index: bool=False) -> Streaks:
        """Perform the streak detection based on prediction. Generate a predicted pattern and
        filter out all the streaks, which pertain to the set of reciprocal lattice points ``hkl``.

        Args:
            hkl : A set of reciprocal lattice points used in the detection.
            hkl_index : Save lattice point indices in generated streaks (:class:`cbclib.Streak`)
                if True.

        Returns:
            New :class:`cbclib.ModelDetector` streak detector with updated ``streaks``.
        """
        return self.model.generate_streaks(hkl, hkl_index=hkl_index)
