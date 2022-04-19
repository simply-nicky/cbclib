from __future__ import annotations
from multiprocessing import cpu_count
from typing import Any, Dict, ItemsView, Iterable, List, Optional, Set, Tuple, Union, ValuesView
from weakref import ref, ReferenceType
import numpy as np
import pandas as pd
from .cxi_protocol import CXIStore
from .data_container import DataContainer, dict_to_object
from .bin import (subtract_background, project_effs, median, median_filter, LSD,
                  maximum_filter, normalize_streak_data, draw_lines, draw_lines_stack,
                  draw_line_indices)
from .cbc_indexing import ScanSetup

Indices = Union[int, slice]

class Transform():
    """Abstract transform class.

    Attributes:
        shape : Data frame shape.

    Raises:
        AttributeError : If shape isn't initialized.
    """
    def __init__(self, shape: Optional[Tuple[int, int]]=None) -> None:
        """
        Args:
            shape : Data frame shape.
        """
        self._shape = shape

    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape

    @shape.setter
    def shape(self, value: Tuple[int, int]):
        if self._shape is None:
            self._shape = value
        else:
            raise ValueError("Shape is already defined.")

    def check_shape(self, shape: Tuple[int, int]) -> bool:
        """Check if shape is equal to the saved shape.

        Args:
            shape : shape to check.

        Returns:
            True if the shapes are equal.
        """
        if self.shape is None:
            self.shape = shape
            return True
        return self.shape == shape

    def forward(self, inp: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def forward_points(self, pts: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, inp: np.ndarray, out: Optional[np.ndarray]=None) -> np.ndarray:
        raise NotImplementedError

    def backward_points(self, pts: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def state_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

class Crop(Transform):
    """Crop transform. Crops a frame according to a region of interest.

    Attributes:
        roi : Region of interest. Comprised of four elements `[y_min, y_max,
            x_min, x_max]`.
        shape : Data frame shape.
    """
    def __init__(self, roi: Iterable[int], shape: Optional[Tuple[int, int]]=None) -> None:
        """
        Args:
            roi : Region of interest. Comprised of four elements `[y_min, y_max,
                x_min, x_max]`.
            shape : Data frame shape.
        """
        super().__init__(shape=shape)
        self.roi = roi

    def forward(self, inp: np.ndarray) -> np.ndarray:
        """Apply the transform to the input.

        Args:
            inp : Input data array.

        Returns:
            Output data array.
        """
        if self.check_shape(inp.shape[-2:]):
            return inp[..., self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]

        raise ValueError(f'input array has invalid shape: {str(inp.shape):s}')

    def forward_points(self, pts: np.ndarray) -> np.ndarray:
        """Apply the transform to a set of points.

        Args:
            pts : Input array of points.

        Returns:
            Output array of points.
        """
        return pts - self.roi[::2]

    def backward(self, inp: np.ndarray, out: Optional[np.ndarray]=None) -> np.ndarray:
        """Tranform back a data array.

        Args:
            inp : Input tranformed data array.
            out : Output data array. A new one created if not prodived.

        Returns:
            Output data array.
        """
        if out is None:
            out = np.zeros(inp.shape[:-2] + self.shape, dtype=inp.dtype)

        if self.check_shape(out.shape[-2:]):
            out[..., self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]] = inp
            return out

        raise ValueError(f'output array has invalid shape: {str(out.shape):s}')

    def backward_points(self, pts: np.ndarray) -> np.ndarray:
        """Tranform back an array of points.

        Args:
            pts : Input tranformed array of points.

        Returns:
            Output array of points.
        """
        return pts + self.roi[::2]

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the transform as a dict.

        Returns:
            A dictionary with all the attributes.
        """
        return {'roi': self.roi[:], 'shape': self.shape}

class Downscale(Transform):
    """Downscale the image by a integer ratio.

    Attributes:
        scale : Downscaling integer ratio.
        shape : Data frame shape.
    """
    def __init__(self, scale: int, shape: Optional[Tuple[int, int]]=None) -> None:
        """
        Args:
            scale : Downscaling integer ratio.
            shape : Data frame shape.
        """
        super().__init__(shape=shape)
        self.scale = scale

    def forward(self, inp: np.ndarray) -> np.ndarray:
        """Apply the transform to the input.

        Args:
            inp : Input data array.

        Returns:
            Output data array.
        """
        if self.check_shape(inp.shape[-2:]):
            return inp[..., ::self.scale, ::self.scale]

        raise ValueError(f'input array has invalid shape: {str(inp.shape):s}')

    def forward_points(self, pts: np.ndarray) -> np.ndarray:
        """Apply the transform to a set of points.

        Args:
            pts : Input array of points.

        Returns:
            Output array of points.
        """
        return pts / self.scale

    def backward(self, inp: np.ndarray, out: Optional[np.ndarray]=None) -> np.ndarray:
        """Tranform back a data array.

        Args:
            inp : Input tranformed data array.
            out : Output data array. A new one created if not prodived.

        Returns:
            Output data array.
        """
        if out is None:
            out = np.empty(inp.shape[:-2] + self.shape, dtype=inp.dtype)

        if self.check_shape(out.shape[-2:]):
            out[...] = np.repeat(np.repeat(inp, self.scale, axis=-2),
                                 self.scale, axis=-1)[..., :self.shape[0], :self.shape[1]]
            return out

        raise ValueError(f'output array has invalid shape: {str(out.shape):s}')

    def backward_points(self, pts: np.ndarray) -> np.ndarray:
        """Tranform back an array of points.

        Args:
            pts : Input tranformed array of points.

        Returns:
            Output array of points.
        """
        return pts * self.scale

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the transform as a dict.

        Returns:
            A dictionary with all the attributes.
        """
        return {'scale': self.scale, 'shape': self.shape}

class Mirror(Transform):
    """Mirror the data around an axis.

    Attributes:
        axis : Axis of reflection.
        shape : Data frame shape.
    """
    def __init__(self, axis: int, shape: Optional[Tuple[int, int]]=None) -> None:
        """
        Args:
            axis : Axis of reflection.
            shape : Data frame shape.
        """
        if axis not in [0, 1]:
            raise ValueError('Axis must equal to 0 or 1')

        super().__init__(shape=shape)
        self.axis = axis

    def forward(self, inp: np.ndarray) -> np.ndarray:
        """Apply the transform to the input.

        Args:
            inp : Input data array.

        Returns:
            Output data array.
        """
        if self.check_shape(inp.shape[-2:]):
            return np.flip(inp, axis=self.axis - 2)

        raise ValueError(f'input array has invalid shape: {str(inp.shape):s}')

    def forward_points(self, pts: np.ndarray) -> np.ndarray:
        """Apply the transform to a set of points.

        Args:
            pts : Input array of points.

        Returns:
            Output array of points.
        """
        pts[:, self.axis] = self.shape[self.axis] - pts[:, self.axis]
        return pts

    def backward(self, inp: np.ndarray, out: Optional[np.ndarray]=None) -> np.ndarray:
        """Tranform back a data array.

        Args:
            inp : Input tranformed data array.
            out : Output data array. A new one created if not prodived.

        Returns:
            Output data array.
        """
        if out is None:
            out = np.empty(inp.shape[:-2] + self.shape, dtype=inp.dtype)

        if self.check_shape(out.shape[-2:]):
            out[...] = self.forward(inp)
            return out

        raise ValueError(f'output array has invalid shape: {str(out.shape):s}')

    def backward_points(self, pts: np.ndarray) -> np.ndarray:
        """Tranform back an array of points.

        Args:
            pts : Input tranformed array of points.

        Returns:
            Output array of points.
        """
        return self.forward_points(pts)

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the transform as a dict.

        Returns:
            A dictionary with all the attributes.
        """
        return {'axis': self.axis, 'shape': self.shape}

class ComposeTransforms(Transform):
    """Composes several transforms together.

    Attributes:
        transforms: List of transforms.
        shape : Data frame shape.
    """
    def __init__(self, transforms: List[Transform], shape: Optional[Tuple[int, int]]=None) -> None:
        """
        Args:
            transforms: List of transforms.
            shape : Data frame shape.
        """
        super().__init__(shape=shape)
        if len(transforms) < 2:
            raise ValueError('Two or more transforms are needed to compose')

        pdict = transforms[0].state_dict()
        pdict['shape'] = self.shape
        self.transforms = [type(transforms[0])(**pdict),]

        for transform in transforms[1:]:
            pdict = transform.state_dict()
            pdict['shape'] = None
            self.transforms.append(type(transform)(**pdict))

    def __iter__(self) -> Iterable:
        return self.transforms.__iter__()

    def __getitem__(self, idx: Indices) -> Transform:
        return self.transforms[idx]

    def forward(self, inp: np.ndarray) -> np.ndarray:
        """Apply the transform to the input.

        Args:
            inp : Input data array.

        Returns:
            Output data array.
        """
        for transform in self:
            inp = transform.forward(inp)
        return inp

    def forward_points(self, pts: np.ndarray) -> np.ndarray:
        """Apply the transform to a set of points.

        Args:
            pts : Input array of points.

        Returns:
            Output array of points.
        """
        for transform in self:
            pts = transform.forward_points(pts)
        return pts

    def backward(self, inp: np.ndarray, out: Optional[np.ndarray]=None) -> np.ndarray:
        """Tranform back a data array.

        Args:
            inp : Input tranformed data array.
            out : Output data array. A new one created if not prodived.

        Returns:
            Output data array.
        """
        for transform in self[1::-1]:
            inp = transform.backward(inp)
        return self[0].backward(inp, out)

    def backward_points(self, pts: np.ndarray) -> np.ndarray:
        """Tranform back an array of points.

        Args:
            pts : Input tranformed array of points.

        Returns:
            Output array of points.
        """
        for transform in self[::-1]:
            pts = transform.backward_points(pts)
        return pts

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the transform as a dict.

        Returns:
            A dictionary with all the attributes.
        """
        return {'transforms': self.transforms, 'shape': self.shape}

class CrystData(DataContainer):
    attr_set: Set[str] = {'input_files'}
    init_set: Set[str] = {'background', 'cor_data', 'data', 'good_frames', 'frames', 'mask',
                          'num_threads', 'output_file', 'streak_data', 'tilts', 'transform',
                          'translations', 'whitefield'}

    # Necessary attributes
    input_files:    CXIStore
    transform:      Transform

    # Automatically generated attributes
    good_frames:    Optional[np.ndarray]
    num_threads:    int
    mask:           Optional[np.ndarray]
    cor_data:       Optional[np.ndarray]
    background:     Optional[np.ndarray]

    # Optional attributes
    data:           Optional[np.ndarray]
    frames:         Optional[np.ndarray]
    output_file:    Optional[CXIStore]
    streak_data:    Optional[np.ndarray]
    tilts:          Optional[np.ndarray]
    translations:   Optional[np.ndarray]
    whitefield:     Optional[np.ndarray]

    def __init__(self, input_files: CXIStore, output_file: Optional[CXIStore]=None,
                 transform: Optional[Transform]=None, **kwargs):
        super(CrystData, self).__init__(input_files=input_files, output_file=output_file,
                                        transform=transform, **kwargs)

        self._init_functions(num_threads=lambda: np.clip(1, 64, cpu_count()))
        if self.shape[0] > 0:
            self._init_functions(good_frames=lambda: np.arange(self.shape[0]))
        if self._isdata:
            self._init_functions(mask=self._mask)
            if self._iswhitefield:
                bgd_func = lambda: project_effs(data=self.data, mask=self.mask,
                                                effs=self.whitefield[None, ...],
                                                num_threads=self.num_threads)
                cor_func = lambda: subtract_background(data=self.data, mask=self.mask,
                                                       bgd=self.background,
                                                       num_threads=self.num_threads)
                self._init_functions(background=bgd_func, cor_data=cor_func)

        self._init_attributes()

    @property
    def _isdata(self) -> bool:
        return self.data is not None

    @property
    def _iswhitefield(self) -> bool:
        return not self.whitefield is None

    @property
    def shape(self) -> Tuple[int, int, int]:
        shape = [0, 0, 0]
        for attr, data in self.items():
            if attr in self.input_files.protocol and data is not None:
                kind = self.input_files.protocol.get_kind(attr)
                if kind == 'sequence':
                    shape[0] = data.shape[0]
                if kind == 'stack':
                    shape[:] = data.shape
                if kind == 'frame':
                    shape[1:] = data.shape
        return tuple(shape)

    def _mask(self) -> np.ndarray:
        mask = np.zeros(self.shape, dtype=bool)
        mask[self.good_frames] = True
        return mask

    def _transform_attribute(self, attr: str, data: np.ndarray, transform: Transform,
                             mode: str='forward') -> np.ndarray:
        kind = self.input_files.protocol.get_kind(attr)
        if kind in ['stack', 'frame']:
            if mode == 'forward':
                data = transform.forward(data)
            elif mode == 'backward':
                data = transform.backward(data)
            else:
                raise ValueError(f'Invalid mode keyword: {mode}')

        return data

    @dict_to_object
    def load(self, attributes: Union[str, List[str], None]=None, idxs: Optional[Iterable[int]]=None,
             processes: int=1, verbose: bool=True) -> CrystData:
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
        with self.input_files:
            self.input_files.update_indices()

            if attributes is None:
                attributes = [attr for attr in self.input_files.keys()
                              if attr in self.init_set]
            else:
                attributes = self.input_files.protocol.str_to_list(attributes)

            if idxs is None:
                idxs = self.input_files.indices()
            data_dict = {'frames': idxs}

            for attr in attributes:
                if attr not in self.input_files.keys():
                    raise ValueError(f"No '{attr}' attribute in the input files")
                if attr not in self.init_set:
                    raise ValueError(f"Invalid attribute: '{attr}'")

                data = self.input_files.load_attribute(attr, idxs, processes, verbose)

                if self.transform and data is not None:
                    data = self._transform_attribute(attr, data, self.transform)

                data_dict[attr] = data

        return data_dict

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
        with self.output_file:
            for attr in self.output_file.protocol.str_to_list(attributes):
                data = self.get(attr)
                if attr in self.output_file.protocol and data is not None:
                    kind = self.output_file.protocol.get_kind(attr)

                    if kind in ['stack', 'sequence']:
                        data = data[self.good_frames]

                    if apply_transform and self.transform:
                        data = self._transform_attribute(attr, data, self.transform,
                                                         mode='backward')

                    self.output_file.save_attribute(attr, np.asarray(data), mode=mode, idxs=idxs)

    @dict_to_object
    def update_output_file(self, output_file: CXIStore) -> CrystData:
        return {'output_file': output_file}

    @dict_to_object
    def clear(self, attributes: Union[str, List[str], None]=None) -> CrystData:
        """Clear the container.

        Args:
            attributes : List of attributes to clear in the container.

        Returns:
            New :class:`CrystData` object with the attributes cleared.
        """
        if attributes is None:
            attributes = self.keys()
        data_dict = {}
        for attr in self.input_files.protocol.str_to_list(attributes):
            data = self.get(attr)
            if attr in self and isinstance(data, np.ndarray):
                data_dict[attr] = None
        return data_dict

    @dict_to_object
    def mask_frames(self, good_frames: Optional[Iterable[int]]=None) -> CrystData:
        """Return a new :class:`CrystData` object with the updated
        good frames mask. Mask empty frames by default.

        Args:
            good_frames : List of good frames' indices. Masks empty
                frames if not provided.

        Returns:
            New :class:`CrystData` object with the updated `good_frames`
            and `whitefield`.
        """
        if good_frames is None:
            good_frames = np.where(self.data.sum(axis=(1, 2)) > 0)[0]
        return {'good_frames': np.asarray(good_frames), 'mask': None}

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
        if self._iswhitefield:
            mat_svd = np.tensordot(self.cor_data, self.cor_data, axes=((1, 2), (1, 2)))
            eig_vals, eig_vecs = np.linalg.eig(mat_svd)
            effs = np.tensordot(eig_vecs, self.cor_data, axes=((0,), (0,)))
            return dict(zip(eig_vals / eig_vals.sum(), effs))

        raise AttributeError("No whitefield in the container")

    @dict_to_object
    def mask_region(self, roi: Iterable[int]) -> CrystData:
        """Return a new :class:`CrystData` object with the updated
        mask. The region defined by the `[y0, y1, x0, 1]` will be masked
        out.

        Args:
            roi : Bad region of interest in the detector plane.
                A set of four coordinates `[y0, y1, x0, y1]`.

        Returns:
            New :class:`CrystData` object with the updated `mask`.
        """
        mask = self.mask.copy()
        mask[self.good_frames, roi[0]:roi[1], roi[2]:roi[3]] = False

        if self._iswhitefield:
            cor_data = self.cor_data.copy()
            cor_data[self.good_frames, roi[0]:roi[1], roi[2]:roi[3]] = 0.0
            return {'cor_data': cor_data, 'mask': mask}

        return {'mask': mask}

    @dict_to_object
    def mask_pupil(self, setup: ScanSetup, padding: int=0) -> CrystData:
        pt0 = setup.kin_to_detector(np.asarray(setup.kin_min))
        pt1 = setup.kin_to_detector(np.asarray(setup.kin_max))
        if self.transform:
            pt0 = self.transform.forward_points(pt0)
            pt1 = self.transform.forward_points(pt1)
        return self.mask_region((int(pt0[0]) - padding, int(pt1[0]) + padding,
                                 int(pt0[1]) - padding, int(pt1[1]) + padding))

    @dict_to_object
    def import_mask(self, mask: np.ndarray, update: str='reset') -> CrystData:
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
        if mask.shape != self.shape[1:]:
            raise ValueError('mask and data have incompatible shapes: '\
                             f'{mask.shape:s} != {self.shape[1:]:s}')

        if update == 'reset':
            return {'mask': mask, 'cor_data': None}
        if update == 'multiply':
            return {'mask': mask * self.mask, 'cor_data': None}

        raise ValueError(f'Invalid update keyword: {update:s}')

    def import_streaks(self, detector: StreakDetector) -> None:
        if detector.parent() is not self:
            raise ValueError("'detector' wasn't derived from this data container")

        self.streak_data = np.zeros(self.shape, dtype=detector.streak_data.dtype)
        self.streak_data[self.good_frames] = detector.streak_data

    @dict_to_object
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
        if whitefield.shape != self.shape[1:]:
            raise ValueError('whitefield and data have incompatible shapes: '\
                             f'{whitefield.shape} != {self.shape[1:]}')
        return {'whitefield': whitefield, 'cor_data': None}

    @dict_to_object
    def update_cor_data(self) -> None:
        return {'cor_data': None}

    @dict_to_object
    def update_background(self, method: str='median', size: int=11,
                          effs: Optional[np.ndarray]=None) -> CrystData:
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
        if self._iswhitefield:
            bgd = self.background.copy()

            if method == 'median':
                outliers = self._mask()
                outliers[self.good_frames] = self.cor_data[self.good_frames] < \
                                             3.0 * np.sqrt(self.background[self.good_frames])
                bgd += median_filter(self.cor_data, size=(size, 1, 1), mask=outliers,
                                     num_threads=self.num_threads)
            elif method == 'pca':
                if effs is None:
                    raise ValueError('No eigen flat fields were provided')

                project_effs(data=self.data, mask=self.mask, effs=effs,
                             out=bgd, num_threads=self.num_threads)

            else:
                raise ValueError('Invalid method argument')

            return {'background': bgd, 'cor_data': None}

        raise AttributeError('No whitefield in the data container')

    @dict_to_object
    def update_mask(self, method: str='perc-bad', pmin: float=0., pmax: float=99.99,
                    vmin: int=0, vmax: int=65535, update: str='reset') -> CrystData:
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
        if update == 'reset':
            data = self.data[self.good_frames]
        elif update == 'multiply':
            data = self.data[self.good_frames] * self.mask[self.good_frames]
        else:
            raise ValueError(f'Invalid update keyword: {update:s}')

        mask = np.zeros(self.shape, dtype=bool)
        if method == 'no-bad':
            mask[self.good_frames] = True
        elif method == 'range-bad':
            mask[self.good_frames] = (data >= vmin) & (data < vmax)
        elif method == 'perc-bad':
            average = median_filter(data, (1, 3, 3), num_threads=self.num_threads)
            offsets = (data.astype(np.int32) - average.astype(np.int32))
            mask[self.good_frames] = (offsets >= np.percentile(offsets, pmin)) & \
                                     (offsets <= np.percentile(offsets, pmax))
        else:
            ValueError('invalid method argument')

        if update == 'reset':
            return {'mask': mask, 'cor_data': None}
        if update == 'multiply':
            return {'mask': mask * self.mask, 'cor_data': None}
        raise ValueError(f'Invalid update keyword: {update:s}')

    @dict_to_object
    def update_transform(self, transform: Transform) -> CrystData:
        """Return a new :class:`CrystData` object with the updated transform
        object.

        Args:
            transform : New :class:`Transform` object.

        Returns:
            New :class:`CrystData` object with the updated transform object.
        """
        data_dict = {'transform': transform}
        for attr, data in self.items():
            if attr in self.input_files.protocol and data is not None:
                data_dict[attr] = self._transform_attribute(attr, data, transform)
        return data_dict

    @dict_to_object
    def update_whitefield(self, method: str='median') -> CrystData:
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
        if method == 'median':
            whitefield = median(data=self.data[self.good_frames], mask=self.mask[self.good_frames],
                                axis=0, num_threads=self.num_threads)
        elif method == 'mean':
            whitefield = np.mean(self.data[self.good_frames] * self.mask[self.good_frames], axis=0)
        else:
            raise ValueError('Invalid method argument')

        return {'whitefield': whitefield, 'cor_data': None}

    def get_detector(self, import_contents: bool=False) -> StreakDetector:
        if self.cor_data is None:
            raise ValueError("No 'cor_data' in the container")

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


class Streaks:
    streak_size: int = 5
    dtype: np.dtype = np.float32
    _indices: Dict[str, int] = {'x0': 0, 'x1': 1, 'y0': 2, 'y1': 3, 'width': 4}

    def __init__(self, array: np.ndarray):
        if array.ndim != 2 or array.shape[1] != self.streak_size:
            raise ValueError(f'Incompatible shape: {array.shape}')

        self._data = np.asarray(array, dtype=self.dtype)

    @classmethod
    def read_dataframe(self, df: pd.DataFrame):
        return self(df[self._indices.keys()])

    def __repr__(self) -> str:
        return dict(self).__repr__()

    def __str__(self) -> str:
        return dict(self).__str__()

    def __len__(self) -> int:
        return self._data.shape[0]

    def __contains__(self, attr: str) -> bool:
        return attr in self.keys()

    def __getitem__(self, attr: str) -> np.ndarray:
        if attr in self:
            return self.__getattribute__(attr)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def keys(self) -> List[str]:
        return [attr for attr, val in vars(Streaks).items() if isinstance(val, property)]

    def values(self) -> ValuesView:
        return dict(self).values()

    def items(self) -> ItemsView:
        return dict(self).items()

    @property
    def x0(self) -> np.ndarray:
        return self._data[:, self._indices['x0']]

    @property
    def x1(self) -> np.ndarray:
        return self._data[:, self._indices['x1']]

    @property
    def y0(self) -> np.ndarray:
        return self._data[:, self._indices['y0']]

    @property
    def y1(self) -> np.ndarray:
        return self._data[:, self._indices['y1']]

    @property
    def width(self) -> np.ndarray:
        return self._data[:, self._indices['width']]

    @property
    def length(self) -> np.ndarray:
        return np.sqrt((self._data[:, 0] - self._data[:, 2])**2 +
                       (self._data[:, 1] - self._data[:, 3])**2)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(dict(self))

    def to_numpy(self) -> np.ndarray:
        return np.copy(self._data)

    def draw(self, shape: Tuple[int, int], max_val: int=1, dilation: float=0.0) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.uint32)
        return draw_lines(mask, self._data, max_val=max_val, dilation=dilation)

    def draw_indices(self, shape: Tuple[int, int], max_val: int=1, dilation: float=0.0) -> pd.DataFrame:
        idxs = draw_line_indices(lines=self._data, shape=shape, max_val=max_val, dilation=dilation)
        return pd.DataFrame({'streaks': idxs[:, 0], 'x': idxs[:, 1], 'y': idxs[:, 2], 'val': idxs[:, 3],
                             'length': self.length[idxs[:, 0]], 'width': self.width[idxs[:, 0]]})

class StreakDetector(DataContainer):
    attr_set: Set[str] = {'parent', 'data', 'frames', 'num_threads'}
    init_set: Set[str] = {'bgd_dilation', 'bgd_mask', 'lsd_obj', 'streak_data', 'streak_dilation',
                          'streak_mask', 'streaks'}
    footprint: np.ndarray = np.array([[[False, False,  True, False, False],
                                       [False,  True,  True,  True, False],
                                       [ True,  True,  True,  True,  True],
                                       [False,  True,  True,  True, False],
                                       [False, False,  True, False, False]]])

    # Necessary attributes
    data:               np.ndarray
    frames:             Dict[int, int]
    num_threads:        int
    parent:             ReferenceType[CrystData]

    # Automatically generated attributes
    lsd_obj:            LSD
    indices:            Dict[int, int]
    streak_width:       Optional[float]
    streak_mask:        Optional[np.ndarray]

    # Optional attributes
    bgd_dilation :      Optional[float]
    bgd_mask :          Optional[np.ndarray]
    streak_data:        Optional[np.ndarray]
    streaks:            Optional[Dict[int, Streaks]]

    def __init__(self, parent: ReferenceType, lsd_obj: Optional[LSD]=None,
                 **kwargs: Union[int, np.ndarray, Dict[int, Streaks]]) -> None:
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
        super(StreakDetector, self).__init__(parent=parent, lsd_obj=lsd_obj, **kwargs)

        self._init_functions(lsd_obj=lambda: LSD(scale=0.9, sigma_scale=0.9, log_eps=0.0,
                                                 ang_th=60.0, density_th=0.5, quant=2e-2),
                             indices=lambda: {frame: index for index, frame in self.frames.items()})
        if self.streaks is not None:
            width_func = lambda: np.mean([val.width.mean() for val in self.streaks.values()])
            self._init_functions(streak_width=width_func, streak_mask=self.draw)

        self._init_attributes()

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.data.shape

    @dict_to_object
    def update_lsd(self, scale: float=0.9, sigma_scale: float=0.9,
                   log_eps: float=0., ang_th: float=60.0, density_th: float=0.5,
                   quant: float=2e-2) -> StreakDetector:
        return {'lsd_obj': LSD(scale=scale, sigma_scale=sigma_scale, log_eps=log_eps,
                               ang_th=ang_th, density_th=density_th, quant=quant)}

    def to_dataframe(self, concatenate: bool=True) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        if self.streaks is None:
            raise ValueError("No 'streaks' specified inside the container.")

        transform = self.parent().transform
        dataframes = []
        for frame, streaks in self.streaks.items():
            df = streaks.to_dataframe()

            if transform:
                df.loc[:, ['y0', 'x0']] = transform.backward_points(df.loc[:, ['y0', 'x0']])
                df.loc[:, ['y1', 'x1']] = transform.backward_points(df.loc[:, ['y1', 'x1']])
            df['streaks'] = df.index
            df['frames'] = frame

            dataframes.append(df)

        if concatenate:
            return pd.concat(dataframes)
        return dataframes

    def export_table(self, dilation: float=0.0, concatenate: bool=True) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        if self.streaks is None:
            raise ValueError("No 'streaks' specified inside the container.")

        transform = self.parent().transform
        dataframes = []
        for index, frame in self.frames.items():
            index = self.indices[frame]
            df = self.streaks[frame].draw_indices(shape=self.shape[1:], dilation=dilation)
            df = df[df['val'] > 0].drop('val', axis=1)
            df['frames'] = frame

            raw_data = self.parent().data[index] * self.parent().mask[index]
            df['I_raw'] = raw_data[df.loc[:, 'y'], df.loc[:, 'x']]
            df['sgn'] = self.parent().cor_data[index][df.loc[:, 'y'], df.loc[:, 'x']]
            df['bgd'] = self.parent().background[index][df.loc[:, 'y'], df.loc[:, 'x']]
            df = df[df.loc[:, 'sgn'] > 0.0]

            if transform:
                df.loc[:, ['y', 'x']] = transform.backward_points(df.loc[:, ['y', 'x']])

            dataframes.append(df)

        if concatenate:
            return pd.concat(dataframes)
        return dataframes

    @dict_to_object
    def generate_streak_data(self, vmin: float, vmax: float,
                             size: Union[Tuple[int, ...], int]=(1, 3, 3)) -> StreakDetector:
        streak_data = median_filter(self.data, size=size, num_threads=self.num_threads)
        streak_data = np.divide(np.clip(streak_data, vmin, vmax) - vmin, vmax - vmin)
        return {'streak_data': streak_data}

    @dict_to_object
    def detect(self, cutoff: float, filter_threshold: float=0.2,
               group_threshold: float=0.6, line_width: float=6.0, n_group: int=2) -> StreakDetector:
        out_dict = self.lsd_obj.detect(self.streak_data, cutoff=cutoff,
                                       filter_threshold=filter_threshold,
                                       group_threshold=group_threshold,
                                       n_group=n_group, dilation=line_width,
                                       num_threads=self.num_threads)
        return {'streaks': {self.frames[idx]: Streaks(np.around(lines[:, :5], 2))
                            for idx, lines in out_dict['lines'].items()}}

    @dict_to_object
    def generate_background_mask(self, bgd_dilation: float=8.0) -> StreakDetector:
        bgd_mask = self.draw(dilation=bgd_dilation)
        return {'bgd_dilation': bgd_dilation, 'bgd_mask': bgd_mask}

    @dict_to_object
    def update_streak_data(self) -> StreakDetector:
        if self.streak_mask is None or self.bgd_mask is None:
            raise AttributeError("'streak_mask' and 'bgd_mask' must be generated before.")

        divisor = self.data
        for _ in range(int(self.streak_width) // 2):
            divisor = maximum_filter(divisor, mask=self.streak_mask, footprint=self.footprint,
                                     num_threads=self.num_threads)

        bgd = self.data * (self.bgd_mask - self.streak_mask)
        for _ in range(int(self.bgd_dilation + self.streak_width) // 2):
            bgd = median_filter(bgd, mask=self.bgd_mask, good_data=bgd,
                                footprint=self.footprint, num_threads=self.num_threads)

        streak_data = normalize_streak_data(data=self.data, bgd=bgd, divisor=divisor,
                                            num_threads=self.num_threads)
        return {'streak_data': streak_data}

    def draw(self, max_val: int=1, dilation: float=0.0) -> np.ndarray:
        if self.streaks is None:
            raise ValueError("No 'streaks' specified inside the container.")

        streaks = {key: val.to_numpy() for key, val in self.streaks.items()}
        mask = np.zeros(self.data.shape, dtype=np.uint32)
        return draw_lines_stack(mask, streaks, max_val=max_val, dilation=dilation,
                                num_threads=self.num_threads)
