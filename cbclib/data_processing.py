from __future__ import annotations
from multiprocessing import cpu_count
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Tuple, TypeVar, Union
from dataclasses import dataclass, field
from weakref import ref, ReferenceType
import numpy as np
import pandas as pd
from .cxi_protocol import CXIStore
from .data_container import DataContainer, ScanSetup, Transform
from .bin import (subtract_background, project_effs, median, median_filter, LSD,
                  maximum_filter, normalize_streak_data, draw_lines, draw_lines_stack,
                  draw_line_indices)

C = TypeVar('C', bound='CrystData')

@dataclass
class CrystData(DataContainer):
    input_file:     CXIStore
    transform:      Transform = None
    num_threads:    int = None
    output_file:    CXIStore = None

    data:           np.ndarray = None
    good_frames:    np.ndarray = None
    mask:           np.ndarray = None
    frames:         np.ndarray = None

    whitefield:     np.ndarray = None
    cor_data:       np.ndarray = None
    background:     np.ndarray = None
    streak_data:    np.ndarray = None

    _no_data_exc: ClassVar[ValueError] = ValueError('No data in the container')
    _no_whitefield_exc: ClassVar[ValueError] = ValueError('No whitefield in the container')

    def __post_init__(self):
        if self.num_threads is None:
            self.num_threads = np.clip(1, 64, cpu_count())

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
        dct = dict(self, **kwargs)
        if isinstance(dct['data'], np.ndarray):
            if isinstance(dct['whitefield'], np.ndarray):
                return CrystDataFull(**dct)
            return CrystDataPart(**dct)
        return CrystData(**dct)

    def backward_points(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.transform:
            return self.transform.backward_points(x, y)
        return x, y

    def forward_points(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.transform:
            return self.transform.forward_points(x, y)
        return x, y

    def load(self: C, attributes: Union[str, List[str], None]=None, idxs: Optional[Iterable[int]]=None,
             processes: int=1, verbose: bool=True) -> C:
        """Load data attributes from the input files in `files` file handler object.

        Args:
            attributes : List of attributes to load. Loads all the data attributes
                contained in the file(s) by default.
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
            data_dict = {'frames' :np.asarray(idxs)}

            for attr in attributes:
                if attr not in self.input_file.keys():
                    raise ValueError(f"No '{attr}' attribute in the input files")
                if attr not in self.keys():
                    raise ValueError(f"Invalid attribute: '{attr}'")

                if self.transform and shape[0] * shape[1]:
                    ss_idxs, fs_idxs = np.indices(shape)
                    ss_idxs, fs_idxs = self.transform.index_array(ss_idxs, fs_idxs)
                    data = self.input_file.load_attribute(attr, idxs=idxs, ss_idxs=ss_idxs, fs_idxs=fs_idxs,
                                                          processes=processes, verbose=verbose)
                else:
                    data = self.input_file.load_attribute(attr, idxs=idxs, processes=processes,
                                                          verbose=verbose)

                data_dict[attr] = data

        return self.replace(**data_dict)

    def save(self, attributes: Union[str, List[str], None]=None, apply_transform: bool=False,
             mode: str='append', idxs: Optional[Iterable[int]]=None) -> None:
        """Save data arrays of the data attributes contained in the container to
        an output file.

        Args:
            attributes : List of attributes to save. Saves all the data attributes
                contained in the container by default.
            apply_transform : Apply `transform` to the data arrays if True.
            mode : Writing mode:

                * `append` : Append the data array to already existing dataset.
                * `insert` : Insert the data under the given indices `idxs`.
                * `overwrite` : Overwrite the existing dataset.

            verbose : Set the verbosity of the loading process.
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
        """Clear the container.

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
        """Return a new :class:`CrystData` object with the new output
        file handler.

        Args:
            output_file : A new output file handler.

        Returns:
            New :class:`CrystData` object with the new output file
            handler.
        """
        return self.replace(output_file=output_file)

    def update_transform(self: C, transform: Transform) -> C:
        """Return a new :class:`CrystData` object with the updated transform
        object.

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

    def mask_frames(self: C, good_frames: Optional[Iterable[int]]=None) -> C:
        """Return a new :class:`CrystData` object with the updated
        good frames mask. Mask empty frames by default.

        Args:
            good_frames : List of good frames' indices. Masks empty
                frames if not provided.

        Returns:
            New :class:`CrystData` object with the updated `good_frames`
            and `whitefield`.
        """
        raise self._no_data_exc

    def mask_region(self: C, roi: Iterable[int]) -> C:
        """Return a new :class:`CrystData` object with the updated mask. The region
        defined by the `[y_min, y_max, x_min, x_max]` will be masked out.

        Args:
            roi : Bad region of interest in the detector plane. A set of four
            coordinates `[y_min, y_max, x_min, x_max]`.

        Returns:
            New :class:`CrystData` object with the updated `mask`.
        """
        raise self._no_data_exc

    def mask_pupil(self: C, setup: ScanSetup, padding: float=0.0) -> C:
        raise self._no_data_exc

    def import_mask(self: C, mask: np.ndarray, update: str='reset') -> C:
        """Return a new :class:`CrystData` object with the new
        mask.

        Args:
            mask : New mask array.
            update : Multiply the new mask and the old one if 'multiply',
                use the new one if 'reset'.

        Raises:
            ValueError : If the mask shape is incompatible with the data.

        Returns:
            New :class:`CrystData` object with the updated `mask`.
        """
        raise self._no_data_exc

    def import_whitefield(self, whitefield: np.ndarray) -> CrystData:
        """Return a new :class:`CrystData` object with the new
        whitefield.

        Args:
            whitefield : New whitefield array.

        Raises:
            ValueError : If the whitefield shape is incompatible with the data.

        Returns:
            New :class:`CrystData` object with the updated `whitefield`.
        """
        raise self._no_data_exc

    def update_mask(self: C, method: str='perc-bad', pmin: float=0., pmax: float=99.99,
                    vmin: int=0, vmax: int=65535, update: str='reset') -> C:
        """Return a new :class:`CrystData` object with the updated
        bad pixels mask.

        Args:
            method : Bad pixels masking methods:

                * 'no-bad' (default) : No bad pixels.
                * 'range-bad' : Mask the pixels which values lie outside
                  of (`vmin`, `vmax`) range.
                * 'perc-bad' : Mask the pixels which values lie outside
                  of the (`pmin`, `pmax`) percentiles.

            vmin : Lower intensity bound of 'range-bad' masking method.
            vmax : Upper intensity bound of 'range-bad' masking method.
            pmin : Lower percentage bound of 'perc-bad' masking method.
            pmax : Upper percentage bound of 'perc-bad' masking method.
            update : Multiply the new mask and the old one if 'multiply',
                use the new one if 'reset'.

        Returns:
            New :class:`CrystData` object with the updated `mask`.
        """
        raise self._no_data_exc

    def update_whitefield(self: C, method: str='median') -> C:
        """Return a new :class:`CrystData` object with new
        whitefield as the median taken through the stack of
        measured frames.

        Args:
            method : Choose between generating a whitefield with the
                help of taking a median ('median') or an average ('mean')
                through a stack of frames.

        Returns:
            New :class:`CrystData` object with the updated `whitefield`.
        """
        raise self._no_data_exc

    def blur_pupil(self: C, setup: ScanSetup, padding: float=0.0, blur: float=0.0) -> C:
        raise self._no_whitefield_exc

    def get_detector(self, import_contents: bool=False) -> StreakDetector:
        raise self._no_whitefield_exc

    def get_pca(self) -> Dict[float, np.ndarray]:
        """Perform the Principal Component Analysis [PCA]_ of the measured data and
        return a set of eigen flatfields (EFF).

        Returns:
            A tuple of ('effs', 'eig_vals'). The elements are
            as follows:

            * 'effs' : Set of eigen flat-fields.
            * 'eig_vals' : Corresponding eigen values for each of the eigen
              flat-fields.

        References:
            .. [PCA] Vincent Van Nieuwenhove, Jan De Beenhouwer, Francesco De Carlo,
                    Lucia Mancini, Federica Marone, and Jan Sijbers, "Dynamic intensity
                    normalization using eigen flat fields in X-ray imaging," Opt.
                    Express 23, 27975-27989 (2015).
        """
        raise self._no_whitefield_exc

    def import_streaks(self, detector: StreakDetector) -> None:
        raise self._no_whitefield_exc

    def update_background(self: C, method: str='median', size: int=11,
                          effs: Optional[np.ndarray]=None) -> C:
        """Return a new :class:`CrystData` object with a new set of flatfields.
        A set of whitefields are generated by the dint of median filtering or Principal
        Component Analysis [PCA]_.

        Args:
            method : Method to generate the flatfields. The following keyword
                values are allowed:

                * 'median' : Median `data` along the first axis.
                * 'pca' : Generate a set of flatfields based on eigen flatfields
                  `effs`. `effs` can be obtained with :func:`CrystData.get_pca` method.

            size : Size of the filter window in pixels used for the 'median' generation
                method.
            effs : Set of Eigen flatfields used for the 'pca' generation method.

        Raises:
            ValueError : If the `method` keyword is invalid.
            AttributeError : If the `whitefield` is absent in the :class:`CrystData`
                container when using the 'pca' generation method.
            ValuerError : If `effs` were not provided when using the 'pca' generation
                method.

        Returns:
            New :class:`CrystData` object with the updated `flatfields`.

        References
        ----------
        .. [PCA] Vincent Van Nieuwenhove, Jan De Beenhouwer, Francesco De Carlo,
                 Lucia Mancini, Federica Marone, and Jan Sijbers, "Dynamic
                 intensity normalization using eigen flat fields in X-ray
                 imaging," Opt. Express 23, 27975-27989 (2015).

        See Also:
            :func:`cbclib.CrystData.get_pca` : Method to generate eigen flatfields.
        """
        raise self._no_whitefield_exc

    def update_cor_data(self: C) -> C:
        raise self._no_whitefield_exc

@dataclass
class CrystDataPart(CrystData):
    input_file:     CXIStore
    transform:      Transform = None
    num_threads:    int = None
    output_file:    CXIStore = None

    data:           np.ndarray = None
    good_frames:    np.ndarray = None
    mask:           np.ndarray = None
    frames:         np.ndarray = None

    whitefield:     np.ndarray = None
    cor_data:       np.ndarray = None
    background:     np.ndarray = None
    streak_data:    np.ndarray = None

    def __post_init__(self):
        super().__post_init__()
        if self.good_frames is None:
            self.good_frames = np.arange(self.shape[0])
        if self.mask is None:
            self.mask = np.ones(self.shape, dtype=bool)

    def mask_frames(self: C, good_frames: Optional[Iterable[int]]=None) -> C:
        if good_frames is None:
            good_frames = np.where(self.data.sum(axis=(1, 2)) > 0)[0]
        return self.replace(good_frames=np.asarray(good_frames))

    def mask_region(self: C, roi: Iterable[int]) -> C:
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
            return self.replace(mask=mask, cor_data=None)
        if update == 'multiply':
            return self.replace(mask=mask * self.mask, cor_data=None)

        raise ValueError(f'Invalid update keyword: {update:s}')

    def import_whitefield(self, whitefield: np.ndarray) -> CrystDataFull:
        if sum(self.shape[1:]) and whitefield.shape != self.shape[1:]:
            raise ValueError('whitefield and data have incompatible shapes: '\
                             f'{whitefield.shape} != {self.shape[1:]}')
        return self.replace(whitefield=whitefield, cor_data=None)

    def update_mask(self, method: str='perc-bad', pmin: float=0., pmax: float=99.99,
                    vmin: int=0, vmax: int=65535, update: str='reset') -> C:
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
            return self.replace(mask=mask, cor_data=None)
        if update == 'multiply':
            return self.replace(mask=mask * self.mask, cor_data=None)
        raise ValueError(f'Invalid update keyword: {update:s}')

    def update_whitefield(self, method: str='median') -> CrystDataFull:
        if method == 'median':
            whitefield = median(self.data[self.good_frames], mask=self.mask[self.good_frames],
                                axis=0, num_threads=self.num_threads)
        elif method == 'mean':
            whitefield = np.mean(self.data[self.good_frames] * self.mask[self.good_frames], axis=0)
        else:
            raise ValueError('Invalid method argument')

        return self.replace(whitefield=whitefield, cor_data=None)

@dataclass
class CrystDataFull(CrystDataPart):
    input_file:     CXIStore
    transform:      Transform = None
    num_threads:    int = None
    output_file:    CXIStore = None

    data:           np.ndarray = None
    good_frames:    np.ndarray = None
    mask:           np.ndarray = None
    frames:         np.ndarray = None

    whitefield:     np.ndarray = None
    cor_data:       np.ndarray = None
    background:     np.ndarray = None
    streak_data:    np.ndarray = None

    def __post_init__(self):
        super().__post_init__()
        if self.background is None:
            self.background = project_effs(self.data, mask=self.mask,
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
        return self.replace(cor_data=self.cor_data * (1.0 - window))

    def get_detector(self, import_contents: bool=False) -> StreakDetector:
        data = np.asarray(self.cor_data[self.good_frames], order='C', dtype=np.float32)
        frames = dict(zip(self.good_frames, self.frames[self.good_frames]))
        if not data.size:
            raise ValueError('No good frames in the stack')

        if import_contents and self.streak_data is not None:
            streak_data = np.asarray(self.streak_data[self.good_frames], order='C',
                                     dtype=np.float32)
            return StreakDetector(parent=ref(self), data=data, frames=frames,
                                  num_threads=self.num_threads, streak_data=streak_data)

        return StreakDetector(parent=ref(self), data=data, frames=frames,
                              num_threads=self.num_threads)

    def get_pca(self) -> Dict[float, np.ndarray]:
        mat_svd = np.tensordot(self.cor_data, self.cor_data, axes=((1, 2), (1, 2)))
        eig_vals, eig_vecs = np.linalg.eig(mat_svd)
        effs = np.tensordot(eig_vecs, self.cor_data, axes=((0,), (0,)))
        return dict(zip(eig_vals / eig_vals.sum(), effs))

    def import_streaks(self, detector: StreakDetector) -> None:
        if detector.parent() is not self:
            raise ValueError("'detector' wasn't derived from this data container")

        self.streak_data = np.zeros(self.shape, dtype=detector.streak_data.dtype)
        self.streak_data[self.good_frames] = detector.streak_data

    def update_background(self, method: str='median', size: int=11,
                          effs: Optional[np.ndarray]=None) -> CrystDataFull:
        bgd = self.background.copy()

        if method == 'median':
            outliers = self.cor_data < 3.0 * np.sqrt(self.background)
            bgd += median_filter(self.cor_data, size=(size, 1, 1), mask=outliers,
                                    num_threads=self.num_threads)
        elif method == 'pca':
            if effs is None:
                raise ValueError('No eigen flat fields were provided')

            project_effs(self.data, mask=self.mask, effs=effs,
                         out=bgd, num_threads=self.num_threads)

        else:
            raise ValueError('Invalid method argument')

        return self.replace(background=bgd, cor_data=None)

    def update_cor_data(self) -> CrystDataFull:
        return self.replace(cor_data=None)

@dataclass
class Streaks(DataContainer):
    x0          : np.ndarray
    y0          : np.ndarray
    x1          : np.ndarray
    y1          : np.ndarray
    width       : np.ndarray
    length      : np.ndarray = field(init=False)
    h           : np.ndarray = None
    k           : np.ndarray = None
    l           : np.ndarray = None
    hkl_index   : np.ndarray = None

    def __post_init__(self):
        self.length = np.sqrt((self.x1 - self.x0)**2 + (self.y1 - self.y0)**2)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(dict(self))

    def to_numpy(self) -> np.ndarray:
        return np.stack((self[attr] for attr in self.contents()), axis=1)

    def pattern_dataframe(self, shape: Tuple[int, int], dp: float=1.0, dilation: float=0.0,
                          profile: str='tophat') -> pd.DataFrame:
        df = pd.DataFrame(self.pattern_dict(shape, dp=dp, dilation=dilation, profile=profile))
        return df[df['p'] > 0.0].drop_duplicates(['x', 'y'])

    def pattern_image(self, shape: Tuple[int, int], dp: float=1e-3, dilation: float=0.0,
                      profile: str='tophat') -> np.ndarray:
        if dp > 1.0 or dp <= 0.0:
            raise ValueError('`dp` must be in the range of (0.0, 1.0]')
        mask = self.pattern_mask(shape, int(1.0 / dp), dilation, profile)
        return mask / int(1.0 / dp)

    def pattern_mask(self, shape: Tuple[int, int], max_val: int=1, dilation: float=0.0,
                     profile: str='tophat') -> np.ndarray:
        mask = np.zeros(shape, dtype=np.uint32)
        return draw_lines(mask, lines=self.to_numpy(), max_val=max_val, dilation=dilation,
                          profile=profile)

    def pattern_dict(self, shape: Tuple[int, int], dp: float=1e-3, dilation: float=0.0,
                       profile: str='tophat') -> Dict[str, np.ndarray]:
        if dp > 1.0 or dp <= 0.0:
            raise ValueError('`dp` must be in the range of (0.0, 1.0]')
        idx, x, y, p = draw_line_indices(lines=self.to_numpy(), shape=shape, max_val=int(1.0 / dp),
                                         dilation=dilation, profile=profile).T
        pattern = {'x': x, 'y': y, 'p': p / int(1.0 / dp)}
        for attr in ['h', 'k', 'l', 'hkl_index']:
            if attr in self.contents():
                pattern[attr] = self[attr][idx]
        return pattern

S = TypeVar('S', bound='StreakDetector')

@dataclass
class StreakDetector(DataContainer):
    data:               np.ndarray
    frames:             Dict[int, int]
    num_threads:        int
    parent:             ReferenceType[CrystData]

    lsd_obj:            LSD = None
    indices:            Dict[int, int] = None
    streak_data:        np.ndarray = None

    streaks:            Dict[int, Streaks] = None
    streak_width:       float = None
    streak_mask:        np.ndarray = None
    bgd_dilation :      float = None
    bgd_mask :          np.ndarray = None

    footprint: ClassVar[np.ndarray] = np.array([[[False, False,  True, False, False],
                                                 [False,  True,  True,  True, False],
                                                 [ True,  True,  True,  True,  True],
                                                 [False,  True,  True,  True, False],
                                                 [False, False,  True, False, False]]])
    _no_streaks_exc: ClassVar[ValueError] = ValueError('No streaks in the container')

    def __post_init__(self) -> None:
        """
        Args:
            parent : The Speckle tracking data container, from which the
                object is derived.
            kwargs : Dictionary of the attributes' data specified in `attr_set`
                and `init_set`.

        Raises:
            ValueError : If an attribute specified in `attr_set` has not been
                provided.
        """
        if self.lsd_obj is None:
            self.update_lsd()
        if self.indices is None:
            self.indices = {frame: index for index, frame in self.frames.items()}

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    def generate_streak_data(self: S, vmin: float, vmax: float,
                             size: Union[Tuple[int, ...], int]=(1, 3, 3)) -> S:
        streak_data = median_filter(self.data, size=size, num_threads=self.num_threads)
        streak_data = np.divide(np.clip(streak_data, vmin, vmax) - vmin, vmax - vmin)
        return self.replace(streak_data=streak_data)

    def update_lsd(self: S, scale: float=0.9, sigma_scale: float=0.9,
                   log_eps: float=0., ang_th: float=60.0, density_th: float=0.5,
                   quant: float=2e-2) -> S:
        return self.replace(lsd_obj=LSD(scale=scale, sigma_scale=sigma_scale,
                                        log_eps=log_eps, ang_th=ang_th,
                                        density_th=density_th, quant=quant))

    def detect(self, cutoff: float, filter_threshold: float, group_threshold: float=0.7,
               dilation: float=0.0, n_group: int=2) -> StreakDetectorFull:
        if self.streak_data is None:
            raise ValueError('No streak data in the container')

        out_dict = self.lsd_obj.detect(self.streak_data, cutoff=cutoff,
                                       filter_threshold=filter_threshold,
                                       group_threshold=group_threshold,
                                       n_group=n_group, dilation=dilation,
                                       num_threads=self.num_threads)
        streaks = {self.frames[idx]: Streaks(*np.around(lines[:, :5], 2).T)
                   for idx, lines in out_dict['lines'].items()}
        return StreakDetectorFull(**dict(self, streaks=streaks))

    def draw(self, max_val: int=1, dilation: float=0.0) -> np.ndarray:
        raise self._no_streaks_exc

    def to_dataframe(self, concatenate: bool=True) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        raise self._no_streaks_exc

    def export_table(self, dilation: float=0.0, profile: str='tophat', concatenate: bool=True) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        raise self._no_streaks_exc

    def generate_bgd_mask(self: S, bgd_dilation: float=8.0) -> S:
        raise self._no_streaks_exc

    def update_streak_data(self: S) -> S:
        raise self._no_streaks_exc

@dataclass
class StreakDetectorFull(StreakDetector):
    data:               np.ndarray
    frames:             Dict[int, int]
    num_threads:        int
    parent:             ReferenceType[CrystData]

    lsd_obj:            LSD = None
    indices:            Dict[int, int] = None
    streak_data:        np.ndarray = None

    streaks:            Dict[int, Streaks] = None
    streak_width:       float = None
    streak_mask:        np.ndarray = None
    bgd_dilation :      float = None
    bgd_mask :          np.ndarray = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.streak_width is None:
            self.streak_width = np.mean([val.width.mean() for val in self.streaks.values()])
        if self.streak_mask is None:
            self.streak_mask = self.draw()

    def draw(self, max_val: int=1, dilation: float=0.0) -> np.ndarray:
        streaks = {key: val.to_numpy() for key, val in self.streaks.items()}
        mask = np.zeros(self.data.shape, dtype=np.uint32)
        return draw_lines_stack(mask, lines=streaks, max_val=max_val, dilation=dilation,
                                num_threads=self.num_threads)

    def to_dataframe(self, concatenate: bool=True) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        dataframes = []
        for frame, streaks in self.streaks.items():
            df = streaks.to_dataframe()

            df.loc[:, 'x0'], df.loc[:, 'y0'] = self.parent().backward_points(df.loc[:, 'x0'], df.loc[:, 'y0'])
            df.loc[:, 'x1'], df.loc[:, 'y1'] = self.parent().backward_points(df.loc[:, 'x1'], df.loc[:, 'y1'])
            df['streaks'] = df.index
            df['frames'] = frame

            dataframes.append(df)

        if concatenate:
            return pd.concat(dataframes)
        return dataframes

    def export_table(self, dilation: float=0.0, profile: str='tophat', concatenate: bool=True) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        dataframes = []
        for index, frame in self.frames.items():
            index = self.indices[frame]
            df = self.streaks[frame].pattern_dataframe(shape=self.shape[1:], dilation=dilation, profile=profile)
            df['frames'] = frame

            raw_data = self.parent().data[index] * self.parent().mask[index]
            df['p'] = self.streak_data[index][df['y'], df['x']]
            df = df[df['p'] > 0.0]
            df['I_raw'] = raw_data[df['y'], df['x']]
            df['sgn'] = self.parent().cor_data[index][df['y'], df['x']]
            df['bgd'] = self.parent().background[index][df['y'], df['x']]
            df['x'], df['y'] = self.parent().backward_points(df['x'], df['y'])

            dataframes.append(df)

        if concatenate:
            return pd.concat(dataframes)
        return dataframes

    def generate_bgd_mask(self, bgd_dilation: float=8.0) -> StreakDetectorFull:
        bgd_mask = self.draw(dilation=bgd_dilation)
        return self.replace(bgd_dilation=bgd_dilation, bgd_mask=bgd_mask)

    def update_streak_data(self) -> StreakDetectorFull:
        divisor = self.data
        for _ in range(int(self.streak_width) // 2):
            divisor = maximum_filter(divisor, mask=self.streak_mask, footprint=self.footprint,
                                     num_threads=self.num_threads)

        if self.bgd_mask is None:
            bgd = np.zeros(self.shape, dtype=self.data.dtype)
        else:
            bgd = self.data * (self.bgd_mask - self.streak_mask)
            for _ in range(int(self.bgd_dilation + self.streak_width) // 2):
                bgd = median_filter(bgd, mask=self.bgd_mask, inp_mask=bgd,
                                    footprint=self.footprint, num_threads=self.num_threads)

        streak_data = normalize_streak_data(self.data, bgd=bgd, divisor=divisor,
                                            num_threads=self.num_threads)
        return self.replace(streak_data=streak_data)
