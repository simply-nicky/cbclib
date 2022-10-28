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
from .cbc_setup import Basis, ScanSamples, ScanSetup, Streaks
from .cbc_indexing import CBDModel
from .cxi_protocol import CXIStore, Indices
from .data_container import DataContainer, Transform, ReferenceType
from .bin import (subtract_background, project_effs, median, median_filter, LSD,
                  maximum_filter, normalise_pattern, draw_line_stack)

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
        cor_data : Background corrected data.
        background : Detector image backgrounds.
        streak_mask : A mask of detected diffraction streaks.
    """
    input_file  : CXIStore
    transform   : Optional[Transform] = None
    num_threads : int = field(default=np.clip(1, 64, cpu_count()))
    output_file : Optional[CXIStore] = None

    data        : Optional[np.ndarray] = None
    good_frames : Optional[np.ndarray] = None
    mask        : Optional[np.ndarray] = None
    frames      : Optional[np.ndarray] = None

    whitefield  : Optional[np.ndarray] = None
    cor_data    : Optional[np.ndarray] = None
    background  : Optional[np.ndarray] = None
    streak_mask : Optional[np.ndarray] = None

    _no_data_exc: ClassVar[ValueError] = ValueError('No data in the container')
    _no_whitefield_exc: ClassVar[ValueError] = ValueError('No whitefield in the container')

    @property
    def shape(self) -> Tuple[int, int, int]:
        shape = [0, 0, 0]
        for attr, data in self.items():
            if attr in self.input_file.protocol and data is not None:
                kind = self.input_file.protocol.get_kind(attr)
                if kind == 'sequence':
                    shape[0] = data.shape[0]
        for attr, data in self.items():
            if attr in self.input_file.protocol and data is not None:
                kind = self.input_file.protocol.get_kind(attr)
                if kind == 'frame':
                    shape[1:] = data.shape
        for attr, data in self.items():
            if attr in self.input_file.protocol and data is not None:
                kind = self.input_file.protocol.get_kind(attr)
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
        if isinstance(dct['data'], np.ndarray):
            if isinstance(dct['whitefield'], np.ndarray):
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
                attributes = self.input_file.protocol.str_to_list(attributes)

            if idxs is None:
                idxs = self.input_file.indices()
            data_dict = {'frames': np.asarray(idxs)}

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

        if attributes is None:
            attributes = list(self.contents())

        with self.input_file:
            shape = self.input_file.read_shape()

        with self.output_file:
            for attr in self.output_file.protocol.str_to_list(attributes):
                data = self.get(attr)
                if attr in self.output_file.protocol and data is not None:
                    kind = self.output_file.protocol.get_kind(attr)

                    if kind in ['stack', 'sequence']:
                        data = data[self.good_frames]

                    if apply_transform and self.transform:
                        if kind in ['stack', 'frame']:
                            out = np.zeros(shape, dtype=data.dtype)
                            data = self.transform.backward(data, out)

                    self.output_file.save_attribute(attr, np.asarray(data), mode=mode, idxs=idxs)

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
        for attr in self.input_file.protocol.str_to_list(attributes):
            if attr not in self.keys():
                raise ValueError(f"Invalid attribute: '{attr}'")

            if isinstance(self[attr], np.ndarray):
                data_dict[attr] = None

        return self.replace(**data_dict)

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
            if attr in self.input_file.protocol and data is not None:
                kind = self.input_file.protocol.get_kind(attr)
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

    def mask_pupil(self: C, setup: ScanSetup, padding: float=0.0) -> C:
        """Return a new :class:`CrystData` object with the pupil region masked.

        Args:
            setup : Experimental setup.
            padding : Pupil region padding in pixels.

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

    def update_mask(self: C, method: str='perc-bad', pmin: float=0., pmax: float=99.99,
                    vmin: int=0, vmax: int=65535, update: str='reset') -> C:
        """Return a new :class:`CrystData` object with the updated bad pixels mask.

        Args:
            method : Bad pixels masking methods. The following keyword values are
                allowed:

                * 'no-bad' (default) : No bad pixels.
                * 'range-bad' : Mask the pixels which values lie outside of (`vmin`,
                  `vmax`) range.
                * 'perc-bad' : Mask the pixels which values lie outside of the (`pmin`,
                  `pmax`) percentiles.

            vmin : Lower intensity bound of 'range-bad' masking method.
            vmax : Upper intensity bound of 'range-bad' masking method.
            pmin : Lower percentage bound of 'perc-bad' masking method.
            pmax : Upper percentage bound of 'perc-bad' masking method.
            update : Multiply the new mask and the old one if 'multiply', use the new
                one if 'reset'.

        Raises:
            ValueError : If there is no ``data`` inside the container.

        Returns:
            New :class:`CrystData` object with the updated ``mask``.
        """
        raise self._no_data_exc

    def update_whitefield(self: C, method: str='median', num_medians: int=5) -> C:
        """Return a new :class:`CrystData` object with new whitefield as the median taken
        through the stack of measured frames.

        Args:
            method : Choose a method to generate a white-field. The following keyboard
                attributes are allowed:

                * `mean` : Taking a mean through the stack of frames.
                * `median` : Taking a median through the stack of frames.
                * `median + mean` : Taking ``num_medians`` medians through subsets of
                  frames and then taking a mean through the stack of medians.

            num_medians : Number of medians to generate for `median + mean` method.

        Raises:
            ValueError : If there is no ``data`` inside the container.

        Returns:
            New :class:`CrystData` object with the updated ``whitefield``.
        """
        raise self._no_data_exc

    def blur_pupil(self: C, setup: ScanSetup, padding: float=0.0, blur: float=0.0) -> C:
        """Blur pupil region in the background corrected images.

        Args:
            setup : Experimental setup.
            padding : Pupil region padding in pixels.
            blur : Blur width in pixels.

        Raises:
            ValueError : If there is no ``whitefield`` inside the container.

        Returns:
            New :class:`CrystData` object with the updated ``cor_data``.
        """
        raise self._no_whitefield_exc

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

        References:
            .. [PCA] Vincent Van Nieuwenhove, Jan De Beenhouwer, Francesco De Carlo, Lucia
                    Mancini, Federica Marone, and Jan Sijbers, "Dynamic intensity
                    normalization using eigen flat fields in X-ray imaging," Opt. Express
                    23, 27975-27989 (2015).
        """
        raise self._no_whitefield_exc

    def import_patterns(self: C, table: pd.DataFrame) -> C:
        """Import a streak mask from a CBC table.

        Args:
            table : CBC table in :class:`pandas.DataFrame` format.

        Returns:
            New container with updated ``streak_mask``.

        See Also:
            cbclib.CBCTable : More info about the CBC table.
        """
        raise self._no_whitefield_exc

    def lsd_detector(self) -> LSDetector:
        """Return a new :class:`cbclib.LSDetector` object based on ``cor_data`` attribute.

        Raises:
            ValueError : If there is no ``whitefield`` inside the container.

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

        Returns:
            A CBC pattern detector based on :class:`cbclib.CBDModel` CBD pattern prediction model.
        """
        raise self._no_whitefield_exc

    def update_background(self: C) -> C:
        """Return a new :class:`CrystData` object with a new set of backgrounds. A set of
        backgrounds is generated by fitting a white-field profile to the measured data.

        Raises:
            ValueError : If there is no ``whitefield`` inside the container.

        Returns:
            New :class:`CrystData` object with the updated ``background``.
        """
        raise self._no_whitefield_exc

    def update_cor_data(self: C) -> C:
        """Return a new :class:`CrystData` object with new background corrected detector
        images.

        Raises:
            ValueError : If there is no ``whitefield`` inside the container.

        Returns:
            New :class:`CrystData` object with the updated ``cor_data``.
        """
        raise self._no_whitefield_exc

@dataclass
class CrystDataPart(CrystData):
    input_file  : CXIStore
    transform   : Optional[Transform] = None
    num_threads : int = field(default=np.clip(1, 64, cpu_count()))
    output_file : Optional[CXIStore] = None

    data        : Optional[np.ndarray] = None
    good_frames : Optional[np.ndarray] = None
    mask        : Optional[np.ndarray] = None
    frames      : Optional[np.ndarray] = None

    whitefield  : Optional[np.ndarray] = None
    cor_data    : Optional[np.ndarray] = None
    background  : Optional[np.ndarray] = None
    streak_mask : Optional[np.ndarray] = None

    def __post_init__(self):
        if self.good_frames is None:
            self.good_frames = np.arange(self.shape[0])
        if self.mask is None:
            self.mask = np.ones(self.shape, dtype=bool)

    def mask_frames(self: C, frames: Optional[Indices]=None) -> C:
        if frames is None:
            frames = np.where(self.data.sum(axis=(1, 2)) > 0)[0]
        return self.replace(good_frames=np.asarray(frames))

    def mask_region(self: C, roi: Indices) -> C:
        mask = self.mask.copy()
        mask[:, roi[0]:roi[1], roi[2]:roi[3]] = False

        if self.cor_data is not None:
            cor_data = self.cor_data.copy()
            cor_data[:, roi[0]:roi[1], roi[2]:roi[3]] = 0.0
            return self.replace(mask=mask, cor_data=cor_data)

        return self.replace(mask=mask)

    def mask_pupil(self: C, setup: ScanSetup, padding: float=0.0) -> C:
        x0, y0 = self.forward_points(*setup.kin_to_detector(np.asarray(setup.kin_min)))
        x1, y1 = self.forward_points(*setup.kin_to_detector(np.asarray(setup.kin_max)))
        return self.mask_region((int(y0 - padding), int(y1 + padding),
                                 int(x0 - padding), int(x1 + padding)))

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

    def update_mask(self, method: str='perc-bad', pmin: float=0., pmax: float=99.99,
                    vmin: int=0, vmax: int=65535, update: str='reset') -> C:
        if vmin >= vmax:
            raise ValueError('vmin must be less than vmax')
        if pmin >= pmax:
            raise ValueError('pmin must be less than pmax')

        if update == 'reset':
            data = self.data
        elif update == 'multiply':
            data = self.data * self.mask
        else:
            raise ValueError(f'Invalid update keyword: {update:s}')

        if method == 'no-bad':
            mask = np.ones(self.shape, dtype=bool)
        elif method == 'range-bad':
            mask = (data >= vmin) & (data < vmax)
        elif method == 'perc-bad':
            average = median_filter(data, (1, 3, 3), num_threads=self.num_threads)
            offsets = (data.astype(np.int32) - average.astype(np.int32))
            mask = (offsets >= np.percentile(offsets, pmin)) & \
                   (offsets <= np.percentile(offsets, pmax))
        else:
            ValueError('invalid method argument')

        if update == 'reset':
            return self.replace(mask=mask)
        if update == 'multiply':
            return self.replace(mask=mask * self.mask)
        raise ValueError(f'Invalid update keyword: {update:s}')

    def update_whitefield(self, method: str='median', num_medians: int=5) -> CrystDataFull:
        if method == 'median':
            whitefield = median(self.data[self.good_frames], mask=self.mask[self.good_frames],
                                axis=0, num_threads=self.num_threads)
        elif method == 'mean':
            whitefield = np.mean(self.data[self.good_frames] * self.mask[self.good_frames], axis=0)
        elif method == 'median + mean':
            data = (self.data[self.good_frames] * self.mask[self.good_frames])
            data = data[:num_medians * (data.shape[0] // num_medians)]
            data = data.reshape((-1, num_medians) + self.shape[-2:])
            whitefield = median(data, axis=0, num_threads=self.num_threads).mean(axis=0)
        else:
            raise ValueError('Invalid method argument')

        return CrystDataFull(**dict(self, whitefield=whitefield))

@dataclass
class CrystDataFull(CrystDataPart):
    input_file  : CXIStore
    transform   : Optional[Transform] = None
    num_threads : int = field(default=np.clip(1, 64, cpu_count()))
    output_file : Optional[CXIStore] = None

    data        : Optional[np.ndarray] = None
    good_frames : Optional[np.ndarray] = None
    mask        : Optional[np.ndarray] = None
    frames      : Optional[np.ndarray] = None

    whitefield  : Optional[np.ndarray] = None
    cor_data    : Optional[np.ndarray] = None
    background  : Optional[np.ndarray] = None
    streak_mask : Optional[np.ndarray] = None

    def __post_init__(self):
        super().__post_init__()
        if self.background is None:
            if self.streak_mask is None:
                mask = self.mask
            else:
                mask = self.mask & np.invert(self.streak_mask)
            self.background = project_effs(self.data, mask=mask,
                                           effs=self.whitefield[None, ...],
                                           num_threads=self.num_threads)
        if self.cor_data is None:
            self.cor_data = subtract_background(self.data, mask=self.mask,
                                                bgd=self.background,
                                                num_threads=self.num_threads)

    def blur_pupil(self, setup: ScanSetup, padding: float=0.0, blur: float=0.0) -> CrystDataFull:
        x0, y0 = self.forward_points(*setup.kin_to_detector(np.asarray(setup.kin_min)))
        x1, y1 = self.forward_points(*setup.kin_to_detector(np.asarray(setup.kin_max)))

        i, j = np.indices(self.shape[1:])
        dtype = self.cor_data.dtype
        window = 0.25 * (np.tanh((i - y0 + padding) / blur, dtype=dtype) + \
                         np.tanh((y1 + padding - i) / blur, dtype=dtype)) * \
                        (np.tanh((j - x0 + padding) / blur, dtype=dtype) + \
                         np.tanh((x1 + padding - j) / blur, dtype=dtype))
        return CrystDataFull(**dict(self, cor_data=self.cor_data * (1.0 - window)))

    def get_pca(self) -> Dict[float, np.ndarray]:
        mat_svd = np.tensordot(self.cor_data, self.cor_data, axes=((1, 2), (1, 2)))
        eig_vals, eig_vecs = np.linalg.eig(mat_svd)
        effs = np.tensordot(eig_vecs, self.cor_data, axes=((0,), (0,)))
        return dict(zip(eig_vals / eig_vals.sum(), effs))

    def import_patterns(self, table: pd.DataFrame) -> CrystDataFull:
        streak_mask = np.zeros(self.shape, dtype=bool)
        for index, frame in zip(self.good_frames, self.frames[self.good_frames]):
            pattern = table[table['frames'] == frame]
            pattern.loc[:, ['x', 'y']] = np.stack(self.forward_points(pattern['x'],
                                                                      pattern['y']), axis=1)
            mask = (0 < pattern['y']) & (pattern['y'] < self.shape[1]) & \
                   (0 < pattern['x']) & (pattern['x'] < self.shape[2])
            pattern = pattern[mask]
            streak_mask[index, pattern['y'], pattern['x']] = True
        return CrystDataFull(**dict(self, streak_mask=streak_mask))

    def lsd_detector(self) -> LSDetector:
        if not self.good_frames.size:
            raise ValueError('No good frames in the stack')

        return LSDetector(parent=ref(self), data=self.cor_data[self.good_frames],
                          frames=self.frames[self.good_frames],
                          num_threads=self.num_threads)

    def model_detector(self, basis: Basis, samples: ScanSamples, setup: ScanSetup) -> ModelDetector:
        if not self.good_frames.size:
            raise ValueError('No good frames in the stack')
        frames = self.frames[self.good_frames]
        models = {idx: CBDModel(basis=basis, sample=samples[frame], setup=setup,
                                transform=self.transform, shape=self.shape[-2:])
                  for idx, frame in enumerate(frames)}

        return ModelDetector(parent=ref(self), data=self.cor_data[self.good_frames],
                             frames=frames, models=models, num_threads=self.num_threads)

    def update_background(self) -> CrystDataFull:
        return CrystDataFull(**dict(self, background=None, cor_data=None))

    def update_cor_data(self) -> CrystDataFull:
        return CrystDataFull(**dict(self, cor_data=None))

D = TypeVar('D', bound='Detector')

class Detector(DataContainer):
    footprint: ClassVar[np.ndarray] = np.array([[[False, False,  True, False, False],
                                                 [False,  True,  True,  True, False],
                                                 [ True,  True,  True,  True,  True],
                                                 [False,  True,  True,  True, False],
                                                 [False, False,  True, False, False]]])
    _no_streaks_exc: ClassVar[ValueError] = ValueError('No streaks in the container')

    data            : np.ndarray
    frames          : np.ndarray
    num_threads     : int
    parent          : ReferenceType[CrystData]
    streaks         : Dict[int, Streaks]

    pattern         : Optional[np.ndarray]
    streak_width    : Optional[float]
    streak_dilation : Optional[float]
    streak_mask     : Optional[np.ndarray]
    bgd_dilation    : Optional[float]
    bgd_mask        : Optional[np.ndarray]

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

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

    def draw(self, max_val: int=1, dilation: float=0.0, profile: str='tophat') -> np.ndarray:
        """Draw a pattern mask by using the detected streaks ``streaks``.

        Args:
            max_val : Maximal mask value
            dilation : Line mask dilation in pixels.
            profile : Line width profiles. The following keyword values are allowed:

                * `tophat` : Top-hat (rectangular) function profile.
                * `linear` : Linear (triangular) function profile.
                * `quad` : Quadratic (parabola) function profile.

        Raises:
            ValueError : If there is no ``streaks`` inside the container.

        Returns:
            A pattern mask.
        """
        raise self._no_streaks_exc

    def draw_streaks(self: D, dilation: float=0.0) -> D:
        """Return a new detector object with updated ``streak_mask``.

        Args:
            dilation : Streak mask dilation in pixels.

        Raises:
            ValueError : If there is no ``streaks`` inside the container.

        Returns:
            A new detector object with udpated ``streak_mask``.
        """
        raise self._no_streaks_exc

    def draw_background(self: D, dilation: float=8.0) -> D:
        """Return a new detector object with updated ``bgd_mask``.

        Args:
            dilation : Background mask dilation in pixels.

        Raises:
            ValueError : If there is no ``streaks`` inside the container.

        Returns:
            A new detector object with udpated ``bgd_mask``.
        """
        raise self._no_streaks_exc

    def export_streaks(self):
        """Export ``streak_mask`` to the parent :class:`cbclib.CrystData` data container.

        Raises:
            ValueError : If there is no ``streaks`` inside the container.
            ValueError : If there is no ``streak_mask`` inside the container.
        """
        raise self._no_streaks_exc

    def export_table(self, dilation: float=0.0, concatenate: bool=True,
                     drop_duplicates: bool=True) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """Export normalised pattern into a :class:`pandas.DataFrame` table.

        Args:
            dilation : Line mask dilation in pixels.
            concatenate : Concatenate sets of patterns for each frame into a single table if
                True.
            drop_duplicates : Discard pixel data with duplicate x and y coordinates if True.

        Raises:
            ValueError : If there is no ``streaks`` inside the container.

        Returns:
            List of :class:`pandas.DataFrame` tables for each frame in ``frames`` if
            ``concatenate`` is False, a single :class:`pandas.DataFrame` otherwise. Table
            contains the following information:

            * `frames` : Frame index.
            * `x`, `y` : Pixel coordinates.
            * `p` : Normalised pattern values.
            * `I_raw` : Measured intensity.
            * `sgn` : Background subtracted intensity.
            * `bgd` : Background values.
        """
        raise self._no_streaks_exc

    def update_pattern(self: D) -> D:
        """Return a new detector object with updated ``pattern``.

        Raises:
            ValueError : If there is no ``streaks`` inside the container.
            ValueError : If there is no ``streak_mask`` inside the container.

        Returns:
            A new detector object with updated ``pattern``.
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
    def __post_init__(self) -> None:
        if self.streak_width is None:
            self.streak_width = np.mean([val.width.mean() for val in self.streaks.values()])

    def draw(self, max_val: int=1, dilation: float=0.0, profile: str='tophat') -> np.ndarray:
        streaks = {key: val.to_numpy() for key, val in self.streaks.items()}
        mask = np.zeros(self.shape, dtype=np.uint32)
        return draw_line_stack(mask, lines=streaks, max_val=max_val, dilation=dilation,
                               profile=profile, num_threads=self.num_threads)

    def draw_background(self: D, dilation: float=8.0) -> D:
        bgd_mask = self.draw(dilation=dilation)
        return self.replace(bgd_dilation=dilation, bgd_mask=bgd_mask)

    def draw_streaks(self: D, dilation: float=0.0) -> D:
        streak_mask = self.draw(dilation=dilation)
        return self.replace(streak_dilation=dilation, streak_mask=streak_mask)

    def export_streaks(self):
        if self.streak_mask is None:
            raise ValueError('No streak_mask in the container')
        if self.parent() is None:
            raise ValueError('Invalid parent: the parent data container was deleted')

        streak_mask = np.zeros(self.parent().shape, dtype=bool)
        streak_mask[self.parent().good_frames] = self.streak_mask
        self.parent().streak_mask = streak_mask

    def export_table(self, dilation: float=0.0, concatenate: bool=True,
                     drop_duplicates: bool=True) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        if self.parent() is None:
            raise ValueError('Invalid parent: the parent data container was deleted')

        dataframes, frames = [], []
        for idx, streaks in self.streaks.items():
            df = streaks.pattern_dataframe(shape=self.shape[1:], dilation=dilation,
                                           drop_duplicates=drop_duplicates)
            df['frames'] = self.frames[idx] * np.ones(df.shape[0], dtype=df['index'].dtype)

            index = self.parent().good_frames[idx]
            raw_data = self.parent().data[index] * self.parent().mask[index]
            df['p'] = self.pattern[idx][df['y'], df['x']]
            df = df[df['p'] > 0.0]
            df['I_raw'] = raw_data[df['y'], df['x']]
            df['sgn'] = self.parent().cor_data[index][df['y'], df['x']]
            df['bgd'] = self.parent().background[index][df['y'], df['x']]
            df['x'], df['y'] = self.parent().backward_points(df['x'], df['y'])

            frames.append(self.frames[idx])
            dataframes.append(df)

        dataframes = [df for _, df in sorted(zip(frames, dataframes))]

        if concatenate:
            return pd.concat(dataframes, ignore_index=True)
        return dataframes

    def update_pattern(self: D) -> D:
        if self.streak_mask is None:
            raise ValueError('No streak_mask in the container')

        divisor = self.data
        for _ in range(int(self.streak_width + self.streak_dilation)):
            divisor = maximum_filter(divisor, mask=self.streak_mask, footprint=self.footprint,
                                     num_threads=self.num_threads)

        if self.bgd_mask is None:
            bgd = np.zeros(self.shape, dtype=self.data.dtype)
        else:
            bgd = np.abs(self.data * np.subtract(self.bgd_mask, self.streak_mask, dtype=int))
            for _ in range(int(self.streak_width + self.bgd_dilation)):
                bgd = median_filter(bgd, mask=self.bgd_mask, inp_mask=bgd,
                                    footprint=self.footprint, num_threads=self.num_threads)

        pattern = normalise_pattern(self.data, bgd=bgd, divisor=divisor,
                                    num_threads=self.num_threads)
        return self.replace(pattern=pattern)

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
        data : Background corrected detector data.
        frames : Frame indices of the detector images.
        num_threads : Number of threads used in the calculations.
        parent : A reference to the parent :class:`cbclib.CrystData` container.
        lsd_obj : a Line Segment Detector object.
        streaks : A dictionary of detected :class:`cbclib.Streaks` streaks.
        pattern : Normalized diffraction patterns.
        streak_width : Average width of detected streaks.
        streak_dilation : A streak mask dilation in pixels.
        streak_mask : A set of streak masks.
        bgd_dilation : A background mask dilation in pixels.
        bgd_mask : A set of background masks.
    """
    data            : np.ndarray
    frames          : np.ndarray
    num_threads     : int
    parent          : ReferenceType[CrystData]
    lsd_obj         : LSD = field(default=LSD(0.9, 0.9, 0.0, 60.0, 0.5, 2e-2))
    streaks         : Dict[int, Streaks] = field(default_factory=dict)

    pattern         : Optional[np.ndarray] = None
    streak_width    : Optional[float] = None
    streak_dilation : Optional[float] = None
    streak_mask     : Optional[np.ndarray] = None
    bgd_dilation    : Optional[float] = None
    bgd_mask        : Optional[np.ndarray] = None

    def detect(self, cutoff: float, filter_threshold: float, group_threshold: float=0.7,
               n_group: int=2, dilation: float=0.0) -> LSDetectorFull:
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
            n_group : Number of grouping iterations to perform.
            dilation : Line mask dilation value in pixels.

        Raises:
            ValueError : If there is no ``pattern`` inside the container.

        Returns:
            A new :class:`LSDetector` container with ``streaks`` updated.
        """
        if self.pattern is None:
            raise ValueError('No pattern in the container')

        out_dict = self.lsd_obj.detect(self.pattern, cutoff=cutoff,
                                       filter_threshold=filter_threshold,
                                       group_threshold=group_threshold,
                                       n_group=n_group, dilation=dilation,
                                       num_threads=self.num_threads)
        streaks = {idx: Streaks(*np.around(lines[:, :5], 2).T)
                   for idx, lines in out_dict['lines'].items() if lines.size}
        idxs = [idx for idx, lines in out_dict['lines'].items() if lines.size]
        return LSDetectorFull(**dict(self.mask_frames(idxs), streaks=streaks))

    def update_lsd(self: D, scale: float=0.9, sigma_scale: float=0.9, log_eps: float=0.0,
                   ang_th: float=60.0, density_th: float=0.5, quant: float=2e-2) -> D:
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

    def generate_pattern(self: D, vmin: float, vmax: float,
                         size: Union[Tuple[int, ...], int]=(1, 3, 3)) -> D:
        """Generate a set of normalised diffraction patterns ``pattern`` based on
        taking a 2D median filter of background corrected detector images ``data`` and
        clipping the values to a (``vmin``, ``vmax``) interval.

        Args:
            vmin : Lower bound of the clipping range.
            vmax : Upper bound of the clipping range.
            size : Size of the median filter footprint.

        Raises:
            ValueError : If ``vmax`` is less than ``vmin``.

        Returns:
            A new :class:`cbclib.LSDetector` container with ``pattern`` updated.
        """
        if vmin >= vmax:
            raise ValueError('vmin must be less than vmax')
        pattern = median_filter(self.data, size=size, num_threads=self.num_threads)
        pattern = np.divide(np.clip(pattern, vmin, vmax) - vmin, vmax - vmin)
        return self.replace(pattern=np.asarray(pattern, dtype=np.float32))

@dataclass
class LSDetectorFull(DetectorFull, LSDetector):
    data            : np.ndarray
    frames          : np.ndarray
    num_threads     : int
    parent          : ReferenceType[CrystData]
    lsd_obj         : LSD = field(default=LSD(0.9, 0.9, 0.0, 60.0, 0.5, 2e-2))
    streaks         : Dict[int, Streaks] = field(default_factory=dict)

    pattern         : Optional[np.ndarray] = None
    streak_width    : Optional[float] = None
    streak_dilation : Optional[float] = None
    streak_mask     : Optional[np.ndarray] = None
    bgd_dilation    : Optional[float] = None
    bgd_mask        : Optional[np.ndarray] = None

@dataclass
class ModelDetector(Detector):
    """A streak detector class based on the CBD pattern prediction. Uses :class:`cbclib.CBDModel` to
    predict a pattern and filters out all the predicted streaks, that correspond to the measured
    intensities above the certain threshold. Provides an interface to generate an indexing tabular
    data.

    Args:
        data : Background corrected detector data.
        frames : Frame indices of the detector images.
        num_threads : Number of threads used in the calculations.
        parent : A reference to the parent :class:`cbclib.CrystData` container.
        models : A dictionary of CBD models.
        streaks : A dictionary of detected :class:`cbclib.Streaks` streaks.
        pattern : Normalized diffraction patterns.
        streak_width : Average width of detected streaks.
        streak_dilation : A streak mask dilation in pixels.
        streak_mask : A set of streak masks.
        bgd_dilation : A background mask dilation in pixels.
        bgd_mask : A set of background masks.
    """
    data            : np.ndarray
    frames          : np.ndarray
    num_threads     : int
    parent          : ReferenceType[CrystData]
    models          : Dict[int, CBDModel]
    streaks         : Dict[int, Streaks] = field(default_factory=dict)

    pattern         : Optional[np.ndarray] = None
    streak_width    : Optional[float] = None
    streak_dilation : Optional[float] = None
    streak_mask     : Optional[np.ndarray] = None
    bgd_dilation    : Optional[float] = None
    bgd_mask        : Optional[np.ndarray] = None

    def detect(self, q_abs: float=0.3, width: float=4.0, threshold: float=0.95,
               dp: float=1e-3) -> ModelDetectorFull:
        """Perform the streak detection based on prediction. Generate a predicted pattern and
        filter out all the streaks, which signal-to-noise ratio is below the ``threshold``.

        Args:
            q_abs : The size of reciprocal space to use in prediction. The reciprocal vectors
                are normalised and span in [0.0 - 1.0] interval.
            width : Difrraction streak width in pixels of a predicted pattern.
            threshold : SNR threshold.
            dp : The quantisation step of a predicted pattern.

        Returns:
            New :class:`cbclib.ModelDetector` streak detector with updated ``streaks``.
        """
        if self.parent() is None:
            raise ValueError('Invalid parent: the parent data container was deleted')

        streaks = {}
        for idx, model in self.models.items():
            hkl = model.generate_hkl(q_abs)
            index = self.parent().good_frames[idx]
            stks = model.filter_streaks(hkl=hkl, signal=self.parent().cor_data[index],
                                        background=self.parent().background[index],
                                        width=width, threshold=threshold, dp=dp,
                                        profile='tophat', num_threads=self.num_threads)
            streaks[idx] = stks
        return ModelDetectorFull(**dict(self, streaks=streaks, streak_width=width))

@dataclass
class ModelDetectorFull(DetectorFull, ModelDetector):
    data            : np.ndarray
    frames          : Dict[int, int]
    num_threads     : int
    parent          : ReferenceType[CrystData]

    models          : Dict[int, CBDModel]
    indices         : Dict[int, int] = field(default_factory=dict)
    streaks         : Dict[int, Streaks] = field(default_factory=dict)

    pattern         : Optional[np.ndarray] = None
    streak_width    : Optional[float] = None
    streak_dilation : Optional[float] = None
    streak_mask     : Optional[np.ndarray] = None
    bgd_dilation    : Optional[float] = None
    bgd_mask        : Optional[np.ndarray] = None
