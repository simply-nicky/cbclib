from __future__ import annotations
from multiprocessing import cpu_count
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from weakref import ref, ReferenceType
import numpy as np
from .cxi_protocol import CXIStore
from .data_container import DataContainer, dict_to_object
from .bin import subtract_background, median, median_filter, LSD, maximum_filter, normalize_streak_data

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

    def integrate(self, axis: int) -> Transform:
        """Return a transform version for the dataset integrated
        along the axis.

        Args:
            axis : Axis of integration.

        Returns:
            A new transform version.
        """
        pdict = self.state_dict()
        if pdict['shape']:
            if axis == 0:
                pdict['shape'] = (1, pdict['shape'][1])
            elif axis == 1:
                pdict['shape'] = (pdict['shape'][0], 1)
            else:
                raise ValueError('Axis must be equal to 0 or 1')

        return type(self)(**pdict)

    def state_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

class Crop(Transform):
    """Crop transform. Crops a frame according to a region of interest.

    Attributes:
        roi : Region of interest. Comprised of four elements `[[y_min, x_min],
            [y_max, x_max]]`.
        shape : Data frame shape.
    """
    def __init__(self, roi: Iterable[int], shape: Optional[Tuple[int, int]]=None) -> None:
        """
        Args:
            roi : Region of interest. Comprised of four elements `[[y_min, x_min],
                [y_max, x_max]]`.
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
            return inp[..., self.roi[0][0]:self.roi[1][0], self.roi[0][1]:self.roi[1][1]]

        raise ValueError(f'input array has invalid shape: {str(inp.shape):s}')

    def forward_points(self, pts: np.ndarray) -> np.ndarray:
        """Apply the transform to a set of points.

        Args:
            pts : Input array of points.

        Returns:
            Output array of points.
        """
        return pts - self.roi[0]

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
            out[..., self.roi[0][0]:self.roi[1][0], self.roi[0][1]:self.roi[1][1]] = inp
            return out

        raise ValueError(f'output array has invalid shape: {str(out.shape):s}')

    def backward_points(self, pts: np.ndarray) -> np.ndarray:
        """Tranform back an array of points.

        Args:
            pts : Input tranformed array of points.

        Returns:
            Output array of points.
        """
        return pts + self.roi[0]

    def integrate(self, axis: int) -> Crop:
        """Return a transform version for the dataset integrated
        along the axis.

        Args:
            axis : Axis of integration.

        Returns:
            A new transform version.
        """
        pdict = self.state_dict()
        pdict['roi'][0][axis] = 0
        pdict['roi'][1][axis] = 1

        if pdict['shape']:
            if axis == 0:
                pdict['shape'] = (1, pdict['shape'][1])
            elif axis == 1:
                pdict['shape'] = (pdict['shape'][0], 1)
            else:
                raise ValueError('Axis must be equal to 0 or 1')

        return Crop(**pdict)

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

    def integrate(self, axis: int) -> ComposeTransforms:
        """Return a transform version for the dataset integrated
        along the axis.

        Args:
            axis : Axis of integration.

        Returns:
            A new transform version.
        """
        pdict = self.state_dict()
        pdict['transforms'] = [transform.integrate(axis) for transform in pdict['transforms']]

        if pdict['shape']:
            if axis == 0:
                pdict['shape'] = (1, pdict['shape'][1])
            elif axis == 1:
                pdict['shape'] = (pdict['shape'][0], 1)
            else:
                raise ValueError('Axis must be equal to 0 or 1')

        return ComposeTransforms(**pdict)

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the transform as a dict.

        Returns:
            A dictionary with all the attributes.
        """
        return {'transforms': self.transforms, 'shape': self.shape}

class CrystData(DataContainer):
    attr_set = {'files'}
    init_set = {'cor_data', 'data', 'flatfields', 'good_frames', 'frames', 'mask',
                'num_threads', 'pupil', 'streak_data', 'streaks', 'tilts', 'transform',
                'translations', 'whitefield'}
    is_points = {'pupil', 'streaks'}

    # Necessary attributes
    files:          CXIStore
    transform:      Transform

    # Automatically generated attributes
    good_frames:    Optional[np.ndarray]
    num_threads:    int
    mask:           Optional[np.ndarray]
    cor_data:       Optional[np.ndarray]

    # Optional attributes
    data:           Optional[np.ndarray]
    flatfields:     Optional[np.ndarray]
    frames:         Optional[np.ndarray]
    pupil:          Optional[np.ndarray]
    streak_data:    Optional[np.ndarray]
    streaks:        Optional[np.ndarray]
    tilts:          Optional[np.ndarray]
    translations:   Optional[np.ndarray]
    whitefield:     Optional[np.ndarray]

    def __init__(self, files: CXIStore, transform: Optional[Transform]=None, **kwargs):
        super(CrystData, self).__init__(files=files, transform=transform, **kwargs)

        self._init_functions(num_threads=lambda: np.clip(1, 64, cpu_count()))
        if self._isdata:
            self._init_functions(good_frames=lambda: np.where(self.data.sum(axis=(1, 2)) > 0)[0],
                                 mask=lambda: np.ones(self.data.shape, dtype=bool),
                                 cor_data=self._cor_data)

        self._init_attributes()

    @property
    def _isdata(self) -> bool:
        return self.data is not None

    def _cor_data(self) -> np.ndarray:
        if self._iswhitefield:
            return subtract_background(data=self.data, mask=self.mask, whitefield=self.whitefield,
                                       flatfields=self.flatfields, num_threads=self.num_threads)
        return None

    def _transform_attribute(self, attr: str, data: np.ndarray, transform: Transform,
                             mode: str='forward') -> np.ndarray:
        kind = self.files.protocol.get_kind(attr)
        if kind in ['stack', 'frame']:
            if mode == 'forward':
                data = transform.forward(data)
            elif mode == 'backward':
                data = transform.backward(data)
            else:
                raise ValueError(f'Invalid mode keyword: {mode}')
        if attr in self.is_points:
            if data.shape[-1] != 2:
                raise ValueError(f"'{attr}' has invalid shape: {str(data.shape)}")

            data = self.transform.forward_points(data)

        return data

    @dict_to_object
    def load(self, attributes: Union[str, List[str], None]=None, indices: Iterable[int]=None,
             processes: int=1, verbose: bool=True) -> CrystData:
        """Load data attributes from the input files in `files` file handler object.

        Args:
            attributes : List of attributes to load. Loads all the data attributes
                contained in the file(s) by default.
            indices : List of frame indices to load.
            processes : Number of parallel workers used during the loading.
            verbose : Set the verbosity of the loading process.

        Raises:
            ValueError : If attribute is not existing in the input file(s).
            ValueError : If attribute is invalid.

        Returns:
            New :class:`CrystData` object with the attributes loaded.
        """
        with self.files:
            self.files.update_indices()

            if attributes is None:
                attributes = [attr for attr in self.files.keys()
                              if attr in self.init_set]
            else:
                attributes = self.files.protocol.str_to_list(attributes)

            if indices is None:
                indices = self.files.indices()
            data_dict = {'frames': indices}

            for attr in attributes:
                if attr not in self.files.keys():
                    raise ValueError(f"No '{attr}' attribute in the input files")
                if attr not in self.init_set:
                    raise ValueError(f"Invalid attribute: '{attr}'")

                data = self.files.load_attribute(attr, indices, processes, verbose)

                if self.transform and data is not None:
                    data = self._transform_attribute(attr, data, self.transform)

                data_dict[attr] = data

        return data_dict

    def save(self, attributes: Union[str, List[str], None]=None, apply_transform=False,
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
        if attributes is None:
            attributes = list(self.contents())
        with self.files:
            for attr in self.files.protocol.str_to_list(attributes):
                data = self.get(attr)
                if attr in self.files.protocol and data is not None:
                    kind = self.files.protocol.get_kind(attr)

                    if kind in ['stack', 'sequence']:
                        data = data[self.good_frames]

                    if apply_transform and self.transform:
                        data = self._transform_attribute(attr, data, self.transform,
                                                         mode='backward')

                    self.files.save_attribute(attr, np.asarray(data), mode=mode, idxs=idxs)

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
        for attr in attributes:
            data = self.get(attr)
            if attr in self.files.protocol and data is not None:
                data_dict[attr] = None
        return data_dict

    @property
    def _iswhitefield(self) -> bool:
        return not self.whitefield is None

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
        return {'good_frames': np.asarray(good_frames)}

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
        if not self._iswhitefield:
            raise AttributeError("No whitefield in the container")

        mat_svd = np.tensordot(self.cor_data, self.cor_data, axes=((1, 2), (1, 2)))
        eig_vals, eig_vecs = np.linalg.eig(mat_svd)
        effs = np.tensordot(eig_vecs, self.cor_data, axes=((0,), (0,)))
        return dict(zip(eig_vals / eig_vals.sum(), effs))

    @dict_to_object
    def mask_region(self, roi: Iterable[int]) -> CrystData:
        """Return a new :class:`CrystData` object with the updated
        mask. The region defined by the `roi` will be masked out.

        Args:
            roi : Bad region of interest in the detector plane.
                Must have a (2, 2) shape.

        Returns:
            New :class:`CrystData` object with the updated `mask`.
        """
        mask = self.mask.copy()
        mask[:, roi[0, 0]:roi[1, 0], roi[0, 1]:roi[1, 1]] = False

        if self._iswhitefield:
            cor_data = self.cor_data.copy()
            cor_data[:, roi[0, 0]:roi[1, 0], roi[0, 1]:roi[1, 1]] = 0.0
            return {'cor_data': cor_data, 'mask': mask}

        return {'mask': mask}

    def mask_pupil(self) -> CrystData:
        if self.pupil is None:
            raise ValueError('pupil is not defined')
        return self.mask_region(self.pupil)

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
        if mask.shape != self.data.shape[1:]:
            raise ValueError('mask and data have incompatible shapes: '\
                             f'{mask.shape:s} != {self.data.shape[1:]:s}')

        if update == 'reset':
            return {'mask': mask, 'cor_data': None}
        if update == 'multiply':
            return {'mask': mask * self.mask, 'cor_data': None}

        raise ValueError(f'Invalid update keyword: {update:s}')

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
        if whitefield.shape != self.data.shape[1:]:
            raise ValueError('whitefield and data have incompatible shapes: '\
                             f'{whitefield.shape:s} != {self.data.shape[1:]:s}')
        return {'whitefield': whitefield, 'cor_data': None}

    @dict_to_object
    def update_cor_data(self) -> None:
        return {'cor_data': None}

    @dict_to_object
    def update_flatfields(self, method: str='median', size: int=11,
                          effs: Optional[np.ndarray]=None) -> CrystData:
        """Return a new :class:`CrystData` object with a new set of flatfields.
        The flatfields are generated by the dint of median filtering or Principal
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

            if method == 'median':
                outliers = np.abs(self.data - self.whitefield) < 3 * np.sqrt(self.whitefield)
                flatfields = median_filter(self.data, size=(size, 1, 1), mask=outliers,
                                           num_threads=self.num_threads)
            elif method == 'pca':
                if effs is None:
                    raise ValueError('No eigen flat fields were provided')

                weights = np.tensordot(self.cor_data, effs, axes=((1, 2), (1, 2))) / \
                          np.sum(effs * effs, axis=(1, 2))
                flatfields = np.tensordot(weights, effs, axes=((1,), (0,))) + self.whitefield
            else:
                raise ValueError('Invalid method argument')

            return {'flatfields': flatfields, 'cor_data': None}

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
            data = self.data
        elif update == 'multiply':
            data = self.data * self.mask
        else:
            raise ValueError(f'Invalid update keyword: {update:s}')

        if method == 'no-bad':
            mask = np.ones(data.shape, dtype=bool)
        elif method == 'range-bad':
            mask = (data >= vmin) & (data < vmax)
        elif method == 'perc-bad':
            offsets = (data.astype(np.int32) -
                       median_filter(data, (1, 3, 3),
                                     num_threads=self.num_threads).astype(np.int32))
            mask = (offsets >= np.percentile(offsets, pmin)) & \
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
            if attr in self.files.protocol and data is not None:
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
            whitefield = median(data=self.data, mask=self.mask, axis=0,
                                num_threads=self.num_threads)
        elif method == 'mean':
            whitefield = np.mean(self.data * self.mask, axis=0)
        else:
            raise ValueError('Invalid method argument')

        return {'whitefield': whitefield, 'cor_data': None}

    def detect_streaks(self, vmin: float, vmax: float,
                       size: Union[Tuple[int, ...], int]=(1, 3, 3)) -> StreakDetector:
        if self.pupil is None or not self._iswhitefield:
            raise ValueError('No pupil and whitefield in the container')

        data = self.cor_data[self.good_frames]
        if not data.size:
            raise ValueError('No good frames in the stack')
        if size is not None:
            data = median_filter(data, size=size, num_threads=self.num_threads)

        streak_data = np.divide(np.clip(data, vmin, vmax) - vmin, vmax - vmin)
        return StreakDetector(parent=ref(self), center=self.pupil.mean(axis=0),
                              frames=self.frames[self.good_frames], data=data,
                              streak_data=streak_data, num_threads=self.num_threads)

class StreakDetector(DataContainer):
    attr_set = {'parent', 'center', 'data', 'frames', 'num_threads', 'streak_data'}
    init_set = {'bgd_dilation', 'bgd_mask', 'lsd_obj', 'streak_dilation', 'streak_mask',
                'streaks'}
    footprint = np.array([[[False, False,  True, False, False],
                           [False,  True,  True,  True, False],
                           [ True,  True,  True,  True,  True],
                           [False,  True,  True,  True, False],
                           [False, False,  True, False, False]]])

    parent: ReferenceType
    lsd_obj: LSD

    bgd_dilation : int
    bgd_mask : np.ndarray
    center: np.ndarray
    data: np.ndarray
    frames: np.ndarray
    num_threads: int
    streak_dilation : int
    streak_data: np.ndarray
    streak_mask: np.ndarray
    streaks: Dict[int, np.ndarray]

    def __init__(self, parent: ReferenceType, lsd_obj: Optional[LSD]=None,
                 **kwargs: Union[int, np.ndarray, Dict[int, np.ndarray]]) -> None:
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
                                                 ang_th=60.0, density_th=0.5, quant=2e-2,
                                                 y_c=self.center[0], x_c=self.center[1]))

        self._init_attributes()

    @dict_to_object
    def update_lsd(self, scale: float=0.9, sigma_scale: float=0.9,
                   log_eps: float=0., ang_th: float=60.0, density_th: float=0.5,
                   quant: float=2e-2) -> StreakDetector:
        return {'lsd_obj': LSD(scale=scale, sigma_scale=sigma_scale, log_eps=log_eps,
                               ang_th=ang_th, density_th=density_th, quant=quant,
                               y_c=self.center[0], x_c=self.center[1])}

    @dict_to_object
    def detect(self, radii: Tuple[float, float, float]=(1.0, 1.0, 1.0),
               threshold: float=1.0, filter_lines: bool=True, n_filter: int=3) -> StreakDetector:
        out_dict = self.lsd_obj.detect(self.streak_data, radii=radii, threshold=threshold,
                                       filter_lines=filter_lines, n_filter=n_filter,
                                       num_threads=self.num_threads)
        return {'streaks': {self.frames[idx]: lines
                            for idx, lines in out_dict['lines'].items()}}

    @dict_to_object
    def update_streak_mask(self, streak_dilation: int=6, bgd_dilation: int=14) -> StreakDetector:
        if self.streaks is None:
            raise AttributeError("'streaks' must be generated before.")
        streak_mask = np.zeros(self.data.shape, dtype=np.uint32)
        streak_mask = self.lsd_obj.draw_lines(streak_mask, self.streaks, self.frames,
                                              dilation=streak_dilation,
                                              num_threads=self.num_threads)
        bgd_mask = np.zeros(self.data.shape, dtype=np.uint32)
        bgd_mask = self.lsd_obj.draw_lines(bgd_mask, self.streaks, self.frames,
                                           dilation=bgd_dilation,
                                           num_threads=self.num_threads)
        return {'bgd_dilation': bgd_dilation, 'bgd_mask': bgd_mask,
                'streak_dilation': streak_dilation, 'streak_mask': streak_mask}

    @dict_to_object
    def update_streak_data(self) -> StreakDetector:
        if self.streak_mask is None:
            raise AttributeError("'streak_mask' must be generated before.")

        divisor = self.data
        for _ in range(self.streak_dilation // 2):
            divisor = maximum_filter(divisor, mask=self.streak_mask, footprint=self.footprint,
                                     num_threads=self.num_threads)

        bgd = self.data * (self.bgd_mask - self.streak_mask)
        for _ in range(self.bgd_dilation // 2):
            bgd = median_filter(bgd, mask=self.bgd_mask, good_data=bgd.astype(bool),
                                footprint=self.footprint, num_threads=self.num_threads)

        streak_data = normalize_streak_data(data=self.data, bgd=bgd, divisor=divisor,
                                            num_threads=self.num_threads)
        return {'streak_data': streak_data}
