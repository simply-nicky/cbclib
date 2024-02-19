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
from typing import Any, ClassVar, List, Optional, Tuple, TypeVar, Union
from dataclasses import dataclass
from weakref import ref
import numpy as np
import pandas as pd
from .cbc_setup import Basis, ScanSamples, ScanSetup, ScanStreaks, CBDModel
from .cxi_protocol import CrystProtocol, CXIStore, FileStore, Indices, Kinds
from .data_container import StringFormatter, DataContainer, Transform, ReferenceType
from .streak_finder import CBSDetector
from .src import median, robust_mean, robust_lsq, Structure

C = TypeVar('C', bound='CrystData')

@dataclass
class CrystData(CrystProtocol, DataContainer):
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
        good_frames : An array of good frames' indices.
        mask : Bad pixels mask.
        frames : List of frame indices inside the container.
        whitefield : Measured frames' white-field.
        snr : Signal-to-noise ratio.
        whitefields : A set of white-fields generated for each pattern separately.
    """
    input_file  : Optional[FileStore] = None
    transform   : Optional[Transform] = None
    num_threads : int = cpu_count()
    output_file : Optional[CXIStore] = None

    _no_data_exc: ClassVar[ValueError] = ValueError('No data in the container')
    _no_whitefield_exc: ClassVar[ValueError] = ValueError('No whitefield in the container')

    @property
    def shape(self) -> Tuple[int, int, int]:
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

    def replace(self: C, **kwargs: Any) -> C:
        """Return a new :class:`cbclib.CrystData` container with replaced data.

        Args:
            kwargs : Replaced attributes.

        Returns:
            A new :class:`cbclib.CrystData` container.
        """
        dct = dict(self, **kwargs)
        if dct['data'] is not None:
            if dct['whitefield'] is not None:
                return CrystDataFull(**dct)
            return CrystDataPart(**dct)
        return CrystData(**dct)

    def backward_points(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transform detector coordinates back.

        Args:
            x : A set of transformed x coordinates.
            y : A set of transformed y coordinates.

        Returns:
            A tuple of x and y coordinates.
        """
        if self.transform:
            return self.transform.backward_points(x, y)
        return x, y

    def clear(self: C, attributes: Union[str, List[str], None]=None) -> C:
        """Clear the data inside the container.

        Args:
            attributes : List of attributes to clear in the container.

        Returns:
            New :class:`CrystData` object with the attributes cleared.
        """
        if attributes is None:
            attributes = self.contents()

        data_dict = dict(self)
        for attr in StringFormatter.str_to_list(attributes):
            if attr not in self.keys():
                raise ValueError(f"Invalid attribute: '{attr}'")

            if isinstance(self[attr], np.ndarray):
                data_dict[attr] = None

        return self.replace(**data_dict)

    def forward_points(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transform detector coordinates.

        Args:
            x : A set of  x coordinates.
            y : A set of y coordinates.

        Returns:
            A tuple of transformed x and y coordinates.
        """
        if self.transform:
            return self.transform.forward_points(x, y)
        return x, y

    def load(self: C, attributes: Union[str, List[str], None]=None, idxs: Optional[Indices]=None,
             processes: int=1, update: bool=True, verbose: bool=True) -> C:
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
        if update:
            self.input_file.update(processes=processes)
        shape = self.input_file.read_frame_shape()

        if attributes is None:
            attributes = [attr for attr in self.input_file.attributes()
                            if attr in self.keys()]
        else:
            attributes = StringFormatter.str_to_list(attributes)

        if idxs is None:
            idxs = np.arange(self.input_file.size)
        else:
            idxs = np.atleast_1d(idxs)
        data_dict = {'frames': idxs}

        for attr in attributes:
            if attr not in self.input_file.attributes():
                raise ValueError(f"No '{attr}' attribute in the input files")
            if attr not in self.keys():
                raise ValueError(f"Invalid attribute: '{attr}'")

            if self.transform and shape[0] * shape[1]:
                ss_idxs, fs_idxs = np.indices(shape)
                ss_idxs, fs_idxs = self.transform.index_array(ss_idxs, fs_idxs)
                data = self.input_file.load(attr, idxs=idxs, ss_idxs=ss_idxs,
                                            fs_idxs=fs_idxs, processes=processes,
                                            verbose=verbose)
            else:
                data = self.input_file.load(attr, idxs=idxs, processes=processes,
                                            verbose=verbose)

            data_dict[attr] = data

        return self.replace(**data_dict)

    def save(self, attributes: Union[str, List[str], None]=None, apply_transform: bool=False,
             mode: str='append', idxs: Optional[Indices]=None) -> None:
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
        if self.output_file is None:
            raise ValueError("'output_file' is not defined inside the container")

        if apply_transform and self.transform:
            if self.input_file is None:
                raise ValueError("'input_file' is not defined inside the container")

            shape = self.input_file.read_frame_shape()

        if attributes is None:
            attributes = list(self.contents())

        for attr in StringFormatter.str_to_list(attributes):
            data = self.get(attr)
            if data is not None:
                kind = self.get_kind(attr)

                if kind in (Kinds.STACK, Kinds.SEQUENCE):
                    data = data[self.good_frames]

                if apply_transform and self.transform:
                    if kind in (Kinds.STACK, Kinds.FRAME):
                        out = np.zeros(data.shape[:-2] + shape, dtype=data.dtype)
                        data = self.transform.backward(data, out)

                self.output_file.save(attr, np.asarray(data), mode=mode, idxs=idxs)

    def update_transform(self: C, transform: Transform) -> C:
        """Return a new :class:`CrystData` object with the updated transform object.

        Args:
            transform : New :class:`Transform` object.

        Returns:
            New :class:`CrystData` object with the updated transform object.
        """
        data_dict = {'transform': transform}

        for attr, data in self.items():
            if data is not None:
                kind = self.get_kind(attr)
                if kind in [Kinds.STACK, Kinds.FRAME]:
                    if self.transform is None:
                        data_dict[attr] = transform.forward(data)
                    else:
                        data_dict[attr] = None

        return self.replace(**data_dict)

    def mask_frames(self: C, frames: Optional[Indices]=None) -> C:
        """Return a new :class:`CrystData` object with the updated good frames mask.
        Mask empty frames by default.

        Args:
            frames : List of good frames' indices. Masks empty frames if not provided.

        Raises:
            ValueError : If there is no ``data`` inside the container.

        Returns:
            New :class:`CrystData` object with the updated ``frames`` and ``whitefield``.
        """
        raise self._no_data_exc

    def mask_region(self: C, roi: Indices) -> C:
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
        raise self._no_data_exc

    def import_mask(self: C, mask: np.ndarray, update: str='reset') -> C:
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
        raise self._no_data_exc

    def reset_mask(self: C) -> C:
        """Reset bad pixel mask. Every pixel is assumed to be good by default.

        Raises:
            ValueError : If there is no ``data`` inside the container.

        Returns:
            New :class:`CrystData` object with the default ``mask``.
        """
        raise self._no_data_exc

    def update_mask(self: C, method: str='no-bad', vmin: int=0, vmax: int=65535,
                    snr_max: float=3.0, roi: Optional[Indices]=None) -> C:
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
        raise self._no_data_exc

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
        raise self._no_data_exc

    def line_detector(self, structure: Structure) -> LineDetector:
        """Return a new :class:`cbclib.LineDetector` object that detects lines in SNR frames.

        Raises:
            ValueError : If there is no ``whitefield`` inside the container.
            ValueError : If there is no ``snr`` inside the container.

        Returns:
            A CBC pattern detector based on :class:`cbclib.bin.LSD` Line Segment Detection [LSD]_
            algorithm.
        """
        raise self._no_whitefield_exc

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
        raise self._no_whitefield_exc

    def update_std(self: C, method="robust-scale", r0: float=0.0, r1: float=0.5,
                   n_iter: int=12, lm: float=9.0) -> C:
        raise self._no_whitefield_exc

    def update_snr(self: C, scales: Optional[np.ndarray]=None) -> C:
        """Return a new :class:`CrystData` object with new background corrected detector
        images.

        Raises:
            ValueError : If there is no ``whitefield`` inside the container.

        Returns:
            New :class:`CrystData` object with the updated ``cor_data``.
        """
        raise self._no_whitefield_exc

    def scale_whitefield(self: C, method: str="robust-lsq", r0: float=0.0, r1: float=0.5,
                         n_iter: int=12, lm: float=9.0) -> C:
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
        raise self._no_whitefield_exc

@dataclass
class CrystDataPart(CrystData):
    def __post_init__(self):
        if self.good_frames is None:
            self.good_frames = np.arange(self.shape[0])
        if self.mask is None:
            self.mask = np.ones(self.shape, dtype=bool)

    def mask_frames(self: C, frames: Optional[Indices]=None) -> C:
        if frames is None:
            frames = np.where(self.data.sum(axis=(1, 2)) > 0)[0]
        return self.replace(good_frames=np.asarray(frames))

    def import_mask(self: C, mask: np.ndarray, update: str='reset') -> C:
        if mask.shape != self.shape[1:]:
            raise ValueError('mask and data have incompatible shapes: '\
                             f'{mask.shape:s} != {self.shape[1:]:s}')

        if update == 'reset':
            return self.replace(mask=mask)
        if update == 'multiply':
            return self.replace(mask=mask * self.mask)

        raise ValueError(f'Invalid update keyword: {update:s}')

    def reset_mask(self: C) -> C:
        return self.replace(mask=np.ones(self.shape[1:], dtype=bool))

    def update_mask(self: C, method: str='no-bad', vmin: int=0, vmax: int=65535,
                    snr_max: float=3.0, roi: Optional[Indices]=None) -> C:
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
        if frames is None:
            frames = np.arange(self.shape[0])

        if method == 'median':
            whitefield = median(inp=self.data[frames] * self.mask, axis=0,
                                num_threads=self.num_threads)
        elif method == 'robust-mean':
            whitefield = robust_mean(inp=self.data[frames] * self.mask,
                                     axis=0, r0=r0, r1=r1, n_iter=n_iter, lm=lm,
                                     num_threads=self.num_threads)
        else:
            raise ValueError('Invalid method argument')

        return CrystDataFull(**dict(self, whitefield=whitefield))

@dataclass
class CrystDataFull(CrystDataPart):
    def __post_init__(self):
        if self.scales is None:
            self.scales = np.ones(self.frames.size)

    def line_detector(self, structure: Structure) -> LineDetector:
        if self.snr is None:
            raise ValueError('No snr in the container')
        if not self.good_frames.size:
            raise ValueError('No good frames in the stack')

        return LineDetector(data=self.snr[self.good_frames], mask=self.mask,
                            parent=ref(self), frames=self.frames[self.good_frames],
                            structure=structure, num_threads=self.num_threads)

    def model_detector(self, basis: Basis, samples: ScanSamples, setup: ScanSetup) -> ModelDetector:
        if self.snr is None:
            raise ValueError('No snr in the container')
        if not self.good_frames.size:
            raise ValueError('No good frames in the stack')

        frames = self.frames[self.good_frames]

        return ModelDetector(data=self.snr[self.good_frames], parent=ref(self), frames=frames,
                             model=CBDModel(basis, samples, setup, num_threads=self.num_threads))

    def update_std(self, method="robust-scale", frames: Optional[Indices]=None,
                   r0: float=0.0, r1: float=0.5, n_iter: int=12, lm: float=9.0) -> CrystDataFull:
        if frames is None:
            frames = np.arange(self.shape[0])

        if method == "robust-scale":
            _, std = robust_mean(inp=self.data[frames] * self.mask,
                                 axis=0, r0=r0, r1=r1, n_iter=n_iter, lm=lm,
                                 num_threads=self.num_threads)
        elif method == "poisson":
            std = np.sqrt(self.whitefield)
        else:
            raise ValueError(f"Invalid method argument: {method}")

        return self.replace(std=std)

    def update_snr(self) -> CrystDataFull:
        whitefields = np.sum(self.scales[..., None, None] * self.whitefield, axis=1)
        return self.replace(snr=np.where(self.std, (self.data * self.mask - whitefields) / self.std, 0.0))

    def scale_whitefield(self, method: str="robust-lsq", r0: float=0.0, r1: float=0.5,
                         n_iter: int=12, lm: float=9.0) -> CrystDataFull:
        y = np.where(self.mask, self.data / self.std, 0.0)[:, self.mask]
        W = np.where(self.mask, self.whitefield / self.std, 0.0)[None, self.mask]

        if method == "robust-lsq":
            scales = robust_lsq(W=W, y=y, axis=1, r0=r0, r1=r1, n_iter=n_iter, lm=lm,
                                num_threads=self.num_threads)
            return self.replace(scales=scales)

        if method == "median":
            scales = median(y * W, axis=1, num_threads=self.num_threads)[:, None] / \
                     median(W * W, axis=1, num_threads=self.num_threads)[:, None]
            return self.replace(scales=scales)

        raise ValueError(f"Invalid method argument: {method}")

class Detector(DataContainer):
    data            : np.ndarray
    frames          : np.ndarray
    num_threads     : int
    parent          : ReferenceType[CrystData]

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    def export_table(self, streaks: ScanStreaks, width: float,
                     kernel: str='rectangular') -> pd.DataFrame:
        """Export normalised pattern into a :class:`pandas.DataFrame` table.

        Args:
            dilation : Line mask dilation in pixels.
            concatenate : Concatenate sets of patterns for each frame into a single table if
                True.

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

        df = streaks.pattern_dataframe(width, kernel=kernel)
        df['snr'] = self.data[df['frames'], df['y'], df['x']]
        df['frames'] = self.frames[df['frames']]
        df['I_raw'] = self.parent().data[df['frames'], df['y'], df['x']]
        df['bgd'] = self.parent().scales[df['frames']] * self.parent().whitefield[df['y'], df['x']]

        return df

@dataclass
class LineDetector(CBSDetector, Detector):
    def __getitem__(self, idxs: Indices) -> LineDetector:
        return self.replace(data=self.data[idxs], mask=self.mask[idxs], frames=self.frames[idxs])

@dataclass
class ModelDetector(Detector):
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

    def __getitem__(self, idxs: Indices) -> ModelDetector:
        return self.replace(data=self.data[idxs], frames=self.frames[idxs], model=self.model[idxs])

    def count_snr(self, streaks: ScanStreaks, hkl: np.ndarray, width: float,
                  kernel: str='rectangular') -> np.ndarray:
        r"""Count the average signal-to-noise ratio for a set of reciprocal lattice points `hkl`.

        Args:
            hkl : Miller indices of reciprocal lattice points.
            width : Diffraction streak width in pixels.

        Returns:
            An array of average SNR values for each reciprocal lattice point in `hkl`.
        """
        snr = np.zeros(hkl.shape[0])
        cnts = np.zeros(hkl.shape[0], dtype=int)
        for idx, stks in streaks.streaks.items():
            df = stks.pattern_dataframe(width=width, shape=self.shape[1:], kernel=kernel)
            np.add.at(snr, df['hkl_id'], self.data[idx, df['y'], df['x']])
            np.add.at(cnts, df['hkl_id'], np.ones(df.shape[0], dtype=int))
        return np.where(cnts, snr / cnts, 0.0)

    def detect(self, hkl: np.ndarray, hkl_index: bool=False) -> ScanStreaks:
        """Perform the streak detection based on prediction. Generate a predicted pattern and
        filter out all the streaks, which pertain to the set of reciprocal lattice points ``hkl``.

        Args:
            hkl : A set of reciprocal lattice points used in the detection.
            hkl_index : Save lattice point indices in generated streaks (:class:`cbclib.Streak`)
                if True.

        Returns:
            New :class:`cbclib.ModelDetector` streak detector with updated ``streaks``.
        """
        return self.model.generate_streaks(hkl, hkl_index)
