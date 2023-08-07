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
from typing import Any, ClassVar, Dict, List, Optional, Tuple, TypeVar, Union
from dataclasses import dataclass, field
from weakref import ref
import numpy as np
import pandas as pd
from .cbc_setup import Basis, ScanSamples, ScanSetup, Streaks, CBDModel
from .cxi_protocol import CXIProtocol, CXIStore, Indices
from .data_container import DataContainer, Transform, ReferenceType
from .bin import (median, robust_mean, robust_lsq, median_filter, LSD, refine_pattern,
                  draw_line_mask, draw_line_image)

C = TypeVar('C', bound='CrystData')

@dataclass
class CrystData(DataContainer):
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
    input_file  : Optional[CXIStore] = None
    transform   : Optional[Transform] = None
    num_threads : int = cpu_count()
    output_file : Optional[CXIStore] = None

    data        : Optional[np.ndarray] = None
    good_frames : Optional[np.ndarray] = None
    mask        : Optional[np.ndarray] = None
    frames      : Optional[np.ndarray] = None

    whitefield  : Optional[np.ndarray] = None
    snr         : Optional[np.ndarray] = None
    whitefields : Optional[np.ndarray] = None

    _no_data_exc: ClassVar[ValueError] = ValueError('No data in the container')
    _no_whitefield_exc: ClassVar[ValueError] = ValueError('No whitefield in the container')

    @property
    def protocol(self) -> Optional[CXIProtocol]:
        if self.input_file is not None:
            return self.input_file.protocol
        if self.output_file is not None:
            return self.output_file.protocol
        return None

    @property
    def shape(self) -> Tuple[int, int, int]:
        shape = [0, 0, 0]
        for attr, data in self.items():
            if attr in self.protocol and data is not None:
                kind = self.protocol.get_kind(attr)
                if kind == 'sequence':
                    shape[0] = data.shape[0]
        for attr, data in self.items():
            if attr in self.protocol and data is not None:
                kind = self.protocol.get_kind(attr)
                if kind == 'frame':
                    shape[1:] = data.shape
        for attr, data in self.items():
            if attr in self.protocol and data is not None:
                kind = self.protocol.get_kind(attr)
                if kind == 'stack':
                    shape[:] = data.shape
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
        for attr in self.protocol.str_to_list(attributes):
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
             processes: int=1, verbose: bool=True) -> C:
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
        with self.input_file:
            self.input_file.update_indices()
            shape = self.input_file.read_shape()

            if attributes is None:
                attributes = [attr for attr in self.input_file.keys()
                              if attr in self.keys()]
            else:
                attributes = self.protocol.str_to_list(attributes)

            if idxs is None:
                idxs = self.input_file.indices()
            else:
                idxs = np.atleast_1d(idxs)
            data_dict = {'frames': idxs}

            for attr in attributes:
                if attr not in self.input_file.keys():
                    raise ValueError(f"No '{attr}' attribute in the input files")
                if attr not in self.keys():
                    raise ValueError(f"Invalid attribute: '{attr}'")

                if self.transform and shape[0] * shape[1]:
                    ss_idxs, fs_idxs = np.indices(shape)
                    ss_idxs, fs_idxs = self.transform.index_array(ss_idxs, fs_idxs)
                    data = self.input_file.load_attribute(attr, idxs=idxs, ss_idxs=ss_idxs,
                                                          fs_idxs=fs_idxs, processes=processes,
                                                          verbose=verbose)
                else:
                    data = self.input_file.load_attribute(attr, idxs=idxs, processes=processes,
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
            with self.input_file:
                shape = self.input_file.read_shape()[-2:]

        if attributes is None:
            attributes = list(self.contents())

        with self.output_file:
            for attr in self.protocol.str_to_list(attributes):
                data = self.get(attr)
                if attr in self.protocol and data is not None:
                    kind = self.protocol.get_kind(attr)

                    if kind in ['stack', 'sequence']:
                        data = data[self.good_frames]

                    if apply_transform and self.transform:
                        if kind in ['stack', 'frame']:
                            out = np.zeros(data.shape[:-2] + shape, dtype=data.dtype)
                            data = self.transform.backward(data, out)

                    self.output_file.save_attribute(attr, np.asarray(data), mode=mode, idxs=idxs)

    def update_output_file(self: C, output_file: CXIStore) -> C:
        """Return a new :class:`CrystData` object with the new output file handler.

        Args:
            output_file : A new output file handler.

        Returns:
            New :class:`CrystData` object with the new output file handler.
        """
        return self.replace(output_file=output_file)

    def update_transform(self: C, transform: Transform) -> C:
        """Return a new :class:`CrystData` object with the updated transform object.

        Args:
            transform : New :class:`Transform` object.

        Returns:
            New :class:`CrystData` object with the updated transform object.
        """
        data_dict = {'transform': transform}

        for attr, data in self.items():
            if attr in self.protocol and data is not None:
                kind = self.protocol.get_kind(attr)
                if kind in ['stack', 'frame']:
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

    def import_whitefield(self: C, whitefield: np.ndarray) -> C:
        """Return a new :class:`CrystData` object with the new
        whitefield.

        Args:
            whitefield : New whitefield array.

        Raises:
            ValueError : If the whitefield shape is incompatible with the data.
            ValueError : If there is no ``data`` inside the container.

        Returns:
            New :class:`CrystData` object with the updated ``whitefield``.
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

    def get_pca(self) -> Dict[float, np.ndarray]:
        """Perform the Principal Component Analysis [PCA]_ of the measured data and return a
        set of eigen flat-fields (EFF).

        Returns:
            A tuple of ('effs', 'eig_vals'). The elements are
            as follows:

            * 'effs' : Set of eigen flat-fields.
            * 'eig_vals' : Corresponding eigen values for each of the eigen flat-fields.

        Raises:
            ValueError : If there is no ``whitefield`` inside the container.
            ValueError : If there is no ``snr`` inside the container.

        References:
            .. [PCA] Vincent Van Nieuwenhove, Jan De Beenhouwer, Francesco De Carlo, Lucia
                    Mancini, Federica Marone, and Jan Sijbers, "Dynamic intensity
                    normalization using eigen flat fields in X-ray imaging," Opt. Express
                    23, 27975-27989 (2015).
        """
        raise self._no_whitefield_exc

    def lsd_detector(self) -> LSDetector:
        """Return a new :class:`cbclib.LSDetector` object based on ``cor_data`` attribute.

        Raises:
            ValueError : If there is no ``whitefield`` inside the container.
            ValueError : If there is no ``snr`` inside the container.

        Returns:
            A CBC pattern detector based on :class:`cbclib.bin.LSD` Line Segment Detection [LSD]_
            algorithm.
        """
        raise self._no_whitefield_exc

    def model_detector(self, basis: Basis, samples: ScanSamples, setup: ScanSetup) -> ModelDetector:
        """Return a new :class:`cbclib.ModelDetector` object based on ``cor_data`` attribute and
        the solution of sample and indexing refinement.

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

    def update_snr(self: C) -> C:
        """Return a new :class:`CrystData` object with new background corrected detector
        images.

        Raises:
            ValueError : If there is no ``whitefield`` inside the container.

        Returns:
            New :class:`CrystData` object with the updated ``cor_data``.
        """
        raise self._no_whitefield_exc

    def update_whitefields(self: C, W: Optional[np.ndarray]=None, r0: float=0.0, r1: float=0.5,
                           n_iter: int=12, lm: float=9.0) -> C:
        """Return a new :class:`CrystData` object with a new set of whitefields. A set of
        backgrounds is generated by robustly fitting a design matrix `W` to the measured
        patterns.

        Args:
            W : Design matrix. `whitefield` is used by default.
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
            New :class:`CrystData` object with the updated ``whitefields``.
        """
        raise self._no_whitefield_exc

@dataclass
class CrystDataPart(CrystData):
    input_file  : Optional[CXIStore] = None
    transform   : Optional[Transform] = None
    num_threads : int = cpu_count()
    output_file : Optional[CXIStore] = None

    data        : Optional[np.ndarray] = None
    good_frames : Optional[np.ndarray] = None
    mask        : Optional[np.ndarray] = None
    frames      : Optional[np.ndarray] = None

    whitefield  : Optional[np.ndarray] = None
    snr         : Optional[np.ndarray] = None
    whitefields : Optional[np.ndarray] = None

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

    def import_whitefield(self, whitefield: np.ndarray) -> CrystDataFull:
        if sum(self.shape[1:]) and whitefield.shape != self.shape[1:]:
            raise ValueError('whitefield and data have incompatible shapes: '\
                             f'{whitefield.shape} != {self.shape[1:]}')
        return CrystDataFull(**dict(self, whitefield=whitefield))

    def reset_mask(self: C) -> C:
        return self.replace(mask=np.ones(self.shape, dtype=bool))

    def update_mask(self: C, method: str='no-bad', vmin: int=0, vmax: int=65535,
                    snr_max: float=3.0, roi: Optional[Indices]=None) -> C:
        if vmin >= vmax:
            raise ValueError('vmin must be less than vmax')
        if roi is None:
            roi = (0, self.shape[1], 0, self.shape[2])

        data = (self.data * self.mask)[:, roi[0]:roi[1], roi[2]:roi[3]]

        if method == 'all-bad':
            mask = np.zeros(data.shape, dtype=bool)
        elif method == 'no-bad':
            mask = np.ones(data.shape, dtype=bool)
        elif method == 'range':
            mask = (data >= vmin) & (data < vmax)
        elif method == 'snr':
            if self.snr is None:
                raise ValueError('No snr in the container')

            snr = self.snr[:, roi[0]:roi[1], roi[2]:roi[3]]
            mask = np.mean(np.abs(snr), axis=0) < snr_max
        else:
            raise ValueError(f'Invalid method argument: {method:s}')

        new_mask = np.copy(self.mask)
        new_mask[:, roi[0]:roi[1], roi[2]:roi[3]] &= mask
        return self.replace(mask=new_mask)

    def update_whitefield(self, method: str='median', frames: Optional[Indices]=None,
                          r0: float=0.0, r1: float=0.5, n_iter: int=12,
                          lm: float=9.0) -> CrystDataFull:
        if frames is None:
            frames = np.arange(self.shape[0])

        if method == 'median':
            whitefield = median(inp=self.data[frames], axis=0,
                                mask=self.mask[frames],
                                num_threads=self.num_threads)
        elif method == 'robust-mean':
            whitefield = robust_mean(inp=self.data[frames] * self.mask[frames],
                                     axis=0, r0=r0, r1=r1, n_iter=n_iter, lm=lm,
                                     num_threads=self.num_threads)
        else:
            raise ValueError('Invalid method argument')

        return CrystDataFull(**dict(self, whitefield=whitefield))

@dataclass
class CrystDataFull(CrystDataPart):
    input_file  : Optional[CXIStore] = None
    transform   : Optional[Transform] = None
    num_threads : int = cpu_count()
    output_file : Optional[CXIStore] = None

    data        : Optional[np.ndarray] = None
    good_frames : Optional[np.ndarray] = None
    mask        : Optional[np.ndarray] = None
    frames      : Optional[np.ndarray] = None

    whitefield  : Optional[np.ndarray] = None
    snr         : Optional[np.ndarray] = None
    whitefields : Optional[np.ndarray] = None

    def get_pca(self) -> Dict[float, np.ndarray]:
        if self.snr is None:
            raise ValueError('No snr in the container')

        mat_svd = np.tensordot(self.snr, self.snr, axes=((1, 2), (1, 2)))
        eig_vals, eig_vecs = np.linalg.eig(mat_svd)
        effs = np.tensordot(eig_vecs, self.snr, axes=((0,), (0,)))
        return dict(zip(eig_vals / eig_vals.sum(), effs))

    def lsd_detector(self) -> LSDetector:
        if self.snr is None:
            raise ValueError('No snr in the container')
        if not self.good_frames.size:
            raise ValueError('No good frames in the stack')

        return LSDetector(snr=self.snr[self.good_frames], parent=ref(self),
                          frames=self.frames[self.good_frames], num_threads=self.num_threads)

    def model_detector(self, basis: Basis, samples: ScanSamples, setup: ScanSetup) -> ModelDetector:
        if self.snr is None:
            raise ValueError('No snr in the container')
        if not self.good_frames.size:
            raise ValueError('No good frames in the stack')

        frames = self.frames[self.good_frames]
        models = {idx: CBDModel(basis=basis, sample=samples[frame], setup=setup,
                                transform=self.transform, shape=self.shape[-2:])
                  for idx, frame in enumerate(frames)}

        return ModelDetector(snr=self.snr[self.good_frames], parent=ref(self),
                             frames=frames, models=models, num_threads=self.num_threads)

    def update_snr(self) -> CrystDataFull:
        if self.whitefields is None:
            std = np.sqrt(self.whitefield)
            snr = np.where(self.whitefield, np.divide(self.data - self.whitefield, std,
                                                      dtype=self.whitefield.dtype), 0.0)
        else:
            std = np.sqrt(self.whitefields)
            snr = np.where(self.whitefields, np.divide(self.data - self.whitefields, std,
                                                       dtype=self.whitefields.dtype), 0.0)
        return self.replace(snr=self.mask * snr)

    def update_whitefields(self, W: Optional[np.ndarray]=None, r0: float=0.0, r1: float=0.5,
                           n_iter: int=12, lm: float=9.0) -> CrystDataFull:
        if W is None:
            W = self.whitefield

        std = np.sqrt(self.whitefield)
        y = np.where(std, np.divide(self.data * self.mask, std, dtype=std.dtype), 0.0)
        W = np.where(std, np.divide(W, std, dtype=std.dtype), 0.0)
        x = robust_lsq(W=W, y=y, axis=(1, 2), r0=r0, r1=r1, n_iter=n_iter, lm=lm,
                       num_threads=self.num_threads)
        return self.replace(whitefields=np.sum(x[..., None, None] * W * std, axis=1))

D = TypeVar('D', bound='Detector')

class Detector(DataContainer):
    profile         : ClassVar[str] = 'gauss'
    _no_streaks_exc : ClassVar[ValueError] = ValueError('No streaks in the container')

    snr             : np.ndarray
    frames          : np.ndarray
    num_threads     : int
    parent          : ReferenceType[CrystData]
    streaks         : Dict[int, Streaks]

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.snr.shape

    def mask_frames(self: D, idxs: Indices) -> D:
        """Choose a subset of frames stored in the container and return a new
        detector object.

        Args:
            idxs : List of frame indices to choose.

        Returns:
            New detector object with the updated ``frames``, ``data``,
            ``streak_data``, and ``streaks``.
        """
        data_dict = {}
        for attr in self.contents():
            if isinstance(self[attr], np.ndarray):
                data_dict[attr] = self[attr][idxs]
        if self.streaks:
            data_dict['streaks'] = {idx: self.streaks[idx] for idx in idxs}
        return self.replace(**data_dict)

    def draw_mask(self, max_val: int=1, dilation: float=0.0, profile: str='tophat') -> np.ndarray:
        """Draw pattern masks by using the detected streaks ``streaks``.

        Args:
            max_val : Maximal mask value
            dilation : Line mask dilation in pixels.
            profile : Line width profiles. The following keyword values are allowed:

                * `tophat` : Top-hat (rectangular) function profile.
                * `linear` : Linear (triangular) function profile.
                * `quad` : Quadratic (parabola) function profile.
                * `gauss` : Gaussian function profile.

        Raises:
            ValueError : If there is no ``streaks`` inside the container.

        Returns:
            A set of pattern masks.
        """
        raise self._no_streaks_exc
    
    def draw_image(self, dilation: float=0.0, profile: str='gauss') -> np.ndarray:
        """Draw pattern images by using the detected streaks ``streaks``.

        Args:
            dilation : Line mask dilation in pixels.
            profile : Line width profiles. The following keyword values are allowed:

                * `tophat` : Top-hat (rectangular) function profile.
                * `linear` : Linear (triangular) function profile.
                * `quad` : Quadratic (parabola) function profile.
                * `gauss` : Gaussian function profile.

        Raises:
            ValueError : If there is no ``streaks`` inside the container.

        Returns:
            A set of pattern images.
        """
        raise self._no_streaks_exc

    def export_table(self, dilation: float=0.0,
                     concatenate: bool=True) -> Union[pd.DataFrame, List[pd.DataFrame]]:
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
        raise self._no_streaks_exc

    def refine_streaks(self: D, dilation: float=0.0) -> D:
        """Refine detected diffraction streaks by fitting a Gaussian across the line.

        Args:
            dilation : Dilation radius in pixels used for the Gaussian fit.

        Returns:
            A new detector with the updated diffraction streaks.
        """
        raise self._no_streaks_exc

    def to_dataframe(self, concatenate: bool=True) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """Export detected streak lines ``streaks`` to a :class:`pandas.DataFrame` table.

        Args:
            concatenate : Concatenate sets of streaks for each frame into a single table if
                True.

        Raises:
            ValueError : If there is no ``streaks`` inside the container.

        Returns:
            List of :class:`pandas.DataFrame` tables for each frame in ``frames`` if
            ``concatenate`` is False, a single :class:`pandas.DataFrame` otherwise. Table
            contains the following information:

            * `frames` : Frame index.
            * `streaks` : Line index.
            * `x0`, `y0`, `x1`, `y1` : Line point coordinates in pixels.
            * `width` : Line width.
            * `length` : Line length.
        """
        raise self._no_streaks_exc

class DetectorFull(Detector):
    def draw_mask(self, max_val: int=1, dilation: float=0.0, profile: str='tophat') -> np.ndarray:
        lines = [streaks.to_lines() for streaks in self.streaks.values()]
        return draw_line_mask(self.shape, lines=lines, max_val=max_val, dilation=dilation,
                              profile=profile, num_threads=self.num_threads)

    def draw_image(self, dilation: float=0.0, profile: str='gauss') -> np.ndarray:
        lines = [streaks.to_lines() for streaks in self.streaks.values()]
        return draw_line_image(self.shape, lines=lines, dilation=dilation,
                               profile=profile, num_threads=self.num_threads)

    def export_table(self, dilation: float=0.0, concatenate: bool=True) -> Union[pd.DataFrame,
                                                                                 List[pd.DataFrame]]:
        if self.parent() is None:
            raise ValueError('Invalid parent: the parent data container was deleted')

        n_streaks = 0
        dataframes = {}
        for idx, streaks in self.streaks.items():
            df = streaks.pattern_dataframe(shape=self.shape[1:], dilation=dilation,
                                           profile=self.profile)

            df['index'] += n_streaks
            df['frames'] = self.frames[idx]
            n_streaks += len(streaks)

            index = self.parent().good_frames[idx]
            df = df[self.parent().mask[index, df['y'], df['x']]]

            df['snr'] = self.snr[idx, df['y'], df['x']]
            df['I_raw'] = self.parent().data[index, df['y'], df['x']]
            df['bgd'] = self.parent().whitefields[index, df['y'], df['x']]
            df['x'], df['y'] = self.parent().backward_points(df['x'], df['y'])

            dataframes[self.frames[idx]] = df

        dataframes = [df for _, df in sorted(dataframes.items())]

        if concatenate:
            return pd.concat(dataframes, ignore_index=True)
        return dataframes

    def refine_streaks(self: D, dilation: float=0.0) -> D:
        lines = {idx: stks.to_lines() for idx, stks in self.streaks.items()}
        lines = refine_pattern(inp=self.snr, lines=lines, dilation=dilation,
                               num_threads=self.num_threads)
        streaks = {idx: self.streaks[idx].replace(x0=lns[:, 0], y0=lns[:, 1], x1=lns[:, 2],
                                                  y1=lns[:, 3], width=lns[:, 4])
                   for idx, lns in lines.items()}
        return self.replace(streaks=streaks)

    def to_dataframe(self, concatenate: bool=True) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        dataframes = []
        for idx, streaks in self.streaks.items():
            df = streaks.to_dataframe()

            df.loc[:, 'x0'], df.loc[:, 'y0'] = self.parent().backward_points(df.loc[:, 'x0'],
                                                                             df.loc[:, 'y0'])
            df.loc[:, 'x1'], df.loc[:, 'y1'] = self.parent().backward_points(df.loc[:, 'x1'],
                                                                             df.loc[:, 'y1'])
            df['index'] = df.index
            df['frames'] = self.frames[idx]

            dataframes.append(df)

        if concatenate:
            return pd.concat(dataframes)
        return dataframes

@dataclass
class LSDetector(Detector):
    """A streak detector class. A class wrapper for streak detection based on Line Segment
    Detector [LSD]_ algorithm. Provides an interface to generate an indexing tabular data.

    Args:
        snr : Signal-to-noise ratio patterns.
        frames : Frame indices of the detector images.
        num_threads : Number of threads used in the calculations.
        parent : A reference to the parent :class:`cbclib.CrystData` container.
        lsd_obj : a Line Segment Detector object.
        streaks : A dictionary of detected :class:`cbclib.Streaks` streaks.
        images : Images used for detection.
    """
    snr             : np.ndarray
    frames          : np.ndarray
    num_threads     : int
    parent          : ReferenceType[CrystData]
    lsd_obj         : LSD = field(default=LSD(0.9, 0.9, 0.0, 45.0, 0.5, 2e-2))
    streaks         : Dict[int, Streaks] = field(default_factory=dict)

    images          : Optional[np.ndarray] = None

    def detect(self, cutoff: float, filter_threshold: float=0.0, group_threshold: float=1.0,
               dilation: float=0.0, profile: str='linear') -> LSDetectorFull:
        """Perform the streak detection. The streak detection comprises three steps: an
        initial LSD detection of lines, a grouping of the detected lines and merging, if
        the normalized cross-correlation value if higher than the ``group_threshold``,
        discarding the lines with a 0-order image moment lower than ``filter_threshold``.

        Args:
            cutoff : Distance cut-off value for lines grouping in pixels.
            filter_threshold : Filtering threshold. A line is discarded if the 0-order image
                moment is lower than ``filter_threshold``.
            group_threshold : Grouping threshold. The lines are merged if the cross-correlation
                value of a pair of lines is higher than ``group_threshold``.
            dilation : Line mask dilation value in pixels.
            profile : Line width profiles. The following keyword values are allowed:

                * `tophat` : Top-hat (rectangular) function profile.
                * `linear` : Linear (triangular) function profile.
                * `quad` : Quadratic (parabola) function profile.
                * `gauss` : Gaussian function profile.

        Raises:
            ValueError : If there is no ``patterns`` inside the container.

        Returns:
            A new :class:`LSDetector` container with ``streaks`` updated.
        """
        if self.images is None:
            raise ValueError('No images in the container')

        out_dict = self.lsd_obj.detect(self.images, cutoff=cutoff,
                                       filter_threshold=filter_threshold,
                                       group_threshold=group_threshold,
                                       dilation=dilation, profile=profile,
                                       num_threads=self.num_threads)
        streaks = {idx: Streaks(*(lines[:, :5].T))
                   for idx, lines in out_dict['lines'].items() if lines.size}
        idxs = [idx for idx, lines in out_dict['lines'].items() if lines.size]
        return LSDetectorFull(**dict(self.mask_frames(idxs), streaks=streaks))

    def update_lsd(self: D, scale: float=0.9, sigma_scale: float=0.9, log_eps: float=0.0,
                   ang_th: float=45.0, density_th: float=0.5, quant: float=2e-2) -> D:
        """Return a new :class:`LSDetector` object with updated :class:`cbclib.bin.LSD` detector.

        Args:
            scale : When different from 1.0, LSD will scale the input image by 'scale' factor
                by Gaussian filtering, before detecting line segments.
            sigma_scale : When ``scale`` is different from 1.0, the sigma of the Gaussian
                filter is :code:`sigma = sigma_scale / scale`, if scale is less than 1.0, and
                :code:`sigma = sigma_scale` otherwise.
            log_eps : Detection threshold, accept if -log10(NFA) > log_eps. The larger the
                value, the more strict the detector is, and will result in less detections.
                The value -log10(NFA) is equivalent but more intuitive than NFA:
                * -1.0 gives an average of 10 false detections on noise.
                *  0.0 gives an average of 1 false detections on noise.
                *  1.0 gives an average of 0.1 false detections on nose.
                *  2.0 gives an average of 0.01 false detections on noise.
            ang_th : Gradient angle tolerance in the region growing algorithm, in degrees.
            density_th : Minimal proportion of 'supporting' points in a rectangle.
            quant : Bound to the quantization error on the gradient norm. Example: if gray
                levels are quantized to integer steps, the gradient (computed by finite
                differences) error due to quantization will be bounded by 2.0, as the worst
                case is when the error are 1 and -1, that gives an error of 2.0.

        Returns:
            A new :class:`LSDetector` with the updated ``lsd_obj``.
        """
        return self.replace(lsd_obj=LSD(scale=scale, sigma_scale=sigma_scale,
                                        log_eps=log_eps, ang_th=ang_th,
                                        density_th=density_th, quant=quant))

    def generate_images(self: D, vmin: float, vmax: float,
                        size: Union[Tuple[int, ...], int]=(1, 3, 3)) -> D:
        """Generate a set of normalised diffraction patterns ``patterns`` based on
        taking a 2D median filter of background corrected detector images ``data`` and
        clipping the values to a (``vmin``, ``vmax``) interval.

        Args:
            vmin : Lower bound of the clipping range.
            vmax : Upper bound of the clipping range.
            size : Size of the median filter footprint.

        Raises:
            ValueError : If ``vmax`` is less than ``vmin``.

        Returns:
            A new :class:`cbclib.LSDetector` container with ``patterns`` updated.
        """
        if vmin >= vmax:
            raise ValueError('vmin must be less than vmax')

        images = median_filter(self.snr, size=size, num_threads=self.num_threads)
        images = np.divide(np.clip(images, vmin, vmax) - vmin, vmax - vmin,
                           dtype=images.dtype)
        return self.replace(images=images)

@dataclass
class LSDetectorFull(DetectorFull, LSDetector):
    snr             : np.ndarray
    frames          : np.ndarray
    num_threads     : int
    parent          : ReferenceType[CrystData]
    lsd_obj         : LSD = field(default=LSD(0.9, 0.9, 0.0, 60.0, 0.5, 2e-2))
    streaks         : Dict[int, Streaks] = field(default_factory=dict)

    images          : Optional[np.ndarray] = None

@dataclass
class ModelDetector(Detector):
    """A streak detector class based on the CBD pattern prediction. Uses :class:`cbclib.CBDModel` to
    predict a pattern and filters out all the predicted streaks, that correspond to the measured
    intensities above the certain threshold. Provides an interface to generate an indexing tabular
    data.

    Args:
        snr : Signal-to-noise ratio patterns.
        frames : Frame indices of the detector images.
        num_threads : Number of threads used in the calculations.
        parent : A reference to the parent :class:`cbclib.CrystData` container.
        models : A dictionary of CBD models.
        streaks : A dictionary of detected :class:`cbclib.Streaks` streaks.
    """
    snr             : np.ndarray
    frames          : np.ndarray
    num_threads     : int
    parent          : ReferenceType[CrystData]
    models          : Dict[int, CBDModel]
    streaks         : Dict[int, Streaks] = field(default_factory=dict)

    def count_snr(self, hkl: np.ndarray, width: float=4.0) -> np.ndarray:
        r"""Count the average signal-to-noise ratio for a set of reciprocal lattice points `hkl`.

        Args:
            hkl : Miller indices of reciprocal lattice points.
            width : Diffraction streak width in pixels.

        Returns:
            An array of average SNR values for each reciprocal lattice point in `hkl`.
        """
        snr = np.zeros(hkl.shape[0])
        cnts = np.zeros(hkl.shape[0], dtype=int)
        for idx, model in self.models.items():
            streaks = model.generate_streaks(hkl, width, hkl_index=True)
            df = streaks.pattern_dataframe(shape=self.shape[1:])
            np.add.at(snr, df['hkl_id'], self.snr[idx, df['y'], df['x']])
            np.add.at(cnts, df['hkl_id'], np.ones(df.shape[0], dtype=int))
        return np.where(cnts, snr / cnts, 0.0)

    def detect(self, hkl: np.ndarray, width: float=4.0, hkl_index: bool=False) -> ModelDetectorFull:
        """Perform the streak detection based on prediction. Generate a predicted pattern and
        filter out all the streaks, which pertain to the set of reciprocal lattice points ``hkl``.

        Args:
            hkl : A set of reciprocal lattice points used in the detection.
            width : Diffraction streak width in pixels of a predicted pattern.
            hkl_index : Save lattice point indices in generated streaks (:class:`cbclib.Streak`)
                if True.

        Returns:
            New :class:`cbclib.ModelDetector` streak detector with updated ``streaks``.
        """
        streaks = {idx: model.generate_streaks(hkl, width, hkl_index=hkl_index)
                   for idx, model in self.models.items()}
        return ModelDetectorFull(**dict(self, streaks=streaks))

@dataclass
class ModelDetectorFull(DetectorFull, ModelDetector):
    snr             : np.ndarray
    frames          : Dict[int, int]
    num_threads     : int
    parent          : ReferenceType[CrystData]

    models          : Dict[int, CBDModel]
    indices         : Dict[int, int] = field(default_factory=dict)
    streaks         : Dict[int, Streaks] = field(default_factory=dict)
