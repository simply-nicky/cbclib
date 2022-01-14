from __future__ import annotations
from multiprocessing import cpu_count
from typing import Dict, Iterable, List, Optional, Tuple, TypeVar, Union
from weakref import ref, ReferenceType
import numpy as np
from .cxi_protocol import CXIStore
from .data_container import DataContainer, dict_to_object
from .bin import subtract_background, median, median_filter, LSD, maximum_filter

class Crop():
    def __init__(self, roi: Iterable[int], shape: Iterable[int]) -> None:
        self.roi, self.shape = roi, shape

    def forward(self, inp: np.ndarray) -> np.ndarray:
        return inp[..., self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]

    def forward_points(self, pts: np.ndarray) -> np.ndarray:
        return pts - self.roi[::2]

    def backward(self, inp: np.ndarray, out: Optional[np.ndarray]=None) -> np.ndarray:
        if out is None:
            out = np.zeros(inp.shape[:-2] + self.shape, dtype=inp.dtype)
        out[..., self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]] = inp
        return out

    def backward_points(self, pts: np.ndarray) -> np.ndarray:
        return pts + self.roi[::2]

class Downscale():
    def __init__(self, scale: int, shape: Iterable[int]) -> None:
        self.scale, self.shape = scale, shape

    def forward(self, inp: np.ndarray) -> np.ndarray:
        return inp[..., ::self.scale, ::self.scale]

    def forward_points(self, pts: np.ndarray) -> np.ndarray:
        return pts / self.scale

    def backward(self, inp: np.ndarray, out: Optional[np.ndarray]=None) -> np.ndarray:
        if out is None:
            out = np.empty(inp.shape[:-2] + self.shape, dtype=inp.dtype)
        out[...] = np.repeat(np.repeat(inp, self.scale, axis=-2),
                                   self.scale, axis=-1)[..., :self.shape[0], :self.shape[1]]
        return out

    def backward_points(self, pts: np.ndarray) -> np.ndarray:
        return pts * self.scale

Transform = TypeVar('Transform', Crop, Downscale)

class ComposeTransforms:
    def __init__(self, transforms: List[Transform]) -> None:
        if len(transforms) < 2:
            raise ValueError('Two or more transforms are needed to compose')
        self.transforms = transforms

    def forward(self, inp: np.ndarray) -> np.ndarray:
        for transform in self.transforms:
            inp = transform.forward(inp)
        return inp

    def forward_points(self, pts: np.ndarray) -> np.ndarray:
        for transform in self.transforms:
            pts = transform.forward_points(pts)
        return pts

    def backward(self, inp: np.ndarray, out: Optional[np.ndarray]=None) -> np.ndarray:
        for transform in self.transforms[1::-1]:
            inp = transform.backward(inp)
        return self.transforms[0].backward(inp, out)

    def backward_points(self, pts: np.ndarray) -> np.ndarray:
        for transform in self.transforms[::-1]:
            pts = transform.backward_points(pts)
        return pts

class CrystData(DataContainer):
    attr_set = {'files'}
    init_set = {'cor_data', 'data', 'flatfields', 'good_frames', 'indices', 'mask',
                'num_threads', 'pupil', 'streak_data', 'streaks', 'tilts', 'transform',
                'translations', 'whitefield'}
    is_points = {'pupil', 'streaks'}

    # Necessary attributes
    files:          CXIStore
    transform:      Transform

    # Automatically generated attributes
    num_threads:    int

    # Optional attributes
    cor_data:       Optional[np.ndarray]
    data:           Optional[np.ndarray]
    flatfields:     Optional[np.ndarray]
    good_frames:    Optional[np.ndarray]
    indices:        Optional[np.ndarray]
    mask:           Optional[np.ndarray]
    pupil:          Optional[np.ndarray]
    streak_data:    Optional[np.ndarray]
    streaks:        Optional[np.ndarray]
    tilts:          Optional[np.ndarray]
    translations:   Optional[np.ndarray]
    whitefield:     Optional[np.ndarray]

    def __init__(self, files: CXIStore, transform: Optional[Transform]=None, **kwargs):
        init_funcs = {'num_threads': lambda: np.clip(1, 64, cpu_count())}
        if kwargs.get('data') is not None:
            init_funcs.update(good_frames=lambda: np.arange(self.data.shape[0]),
                              mask=self._generate_mask,
                              cor_data=self.update_cor_data.inplace_update)

        super(CrystData, self).__init__(init_funcs=init_funcs, files=files,
                                        transform=transform, **kwargs)

    def _generate_mask(self) -> np.ndarray:
        mask = np.ones(self.data.shape, dtype=bool)
        if self.pupil is not None:
            mask[:, self.pupil[0]:self.pupil[1], self.pupil[2]:self.pupil[3]] = False
        return mask

    @dict_to_object
    def load(self, attributes: Union[str, List[str]], indices: Iterable[int]=None,
             processes: int=1, verbose: bool=True) -> None:
        if indices is None:
            indices = self.files.indices()
        data_dict = {'indices': indices}

        for attr in self.files.protocol.str_to_list(attributes):
            if attr not in self.files.keys():
                raise ValueError(f"No '{attr}' attribute in the input files")
            if attr not in self.init_set:
                raise ValueError(f"Invalid attribute: '{attr}'")

            kind = self.files.protocol.get_kind(attr)
            data = self.files.load_attribute(attr, indices, processes, verbose)

            if self.transform:
                if kind in ['stack', 'frame']:
                    data = self.transform.forward(data)
                if attr in self.is_points:
                    if data.shape[-1] != 2:
                        raise ValueError(f"'{attr}' data has invalid shape: {str(data.shape)}")
                    data = self.transform.forward_points(data)

            data_dict[attr] = data

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

        Returns:
            New :class:`CrystData` object with the updated `mask`.
        """
        mask = self.mask.copy()
        mask[:, roi[0]:roi[1], roi[2]:roi[3]] = False

        if self._iswhitefield:
            cor_data = self.cor_data.copy()
            cor_data[:, roi[0]:roi[1], roi[2]:roi[3]] = 0
            return {'cor_data': cor_data, 'mask': mask}

        return {'mask': mask}

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

        if self.pupil is not None:
            mask[:, self.pupil[0]:self.pupil[1], self.pupil[2]:self.pupil[3]] = False

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
        if self._iswhitefield:
            cor_data = subtract_background(data=self.data, mask=self.mask,
                                           whitefield=self.whitefield,
                                           flatfields=self.flatfields,
                                           num_threads=self.num_threads)
            return {'cor_data': cor_data}

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

        if self.pupil is not None:
            mask[:, self.pupil[0]:self.pupil[1], self.pupil[2]:self.pupil[3]] = False

        if update == 'reset':
            return {'mask': mask, 'cor_data': None}
        if update == 'multiply':
            return {'mask': mask * self.mask, 'cor_data': None}
        raise ValueError(f'Invalid update keyword: {update:s}')

    @dict_to_object
    def update_transform(self, transform: Transform) -> CrystData:
        return {'transform': transform}

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

        data = self.cor_data
        if size is not None:
            data = median_filter(data, size=size, num_threads=self.num_threads)

        streak_data = np.divide(np.clip(data, vmin, vmax) - vmin, vmax - vmin)
        return StreakDetector(parent=ref(self), center=self.pupil.mean(axis=0),
                              indices=self.indices, data=data, streak_data=streak_data,
                              num_threads=self.num_threads)

class StreakDetector(DataContainer):
    attr_set = {'parent', 'center', 'data', 'indices', 'num_threads', 'streak_data'}
    init_set = {'lsd_obj', 'streak_mask', 'streaks'}
    footprint = np.array([[[False, False,  True, False, False],
                           [False,  True,  True,  True, False],
                           [ True,  True,  True,  True,  True],
                           [False,  True,  True,  True, False],
                           [False, False,  True, False, False]]])

    parent: ReferenceType
    lsd_obj: LSD

    center: np.ndarray
    data: np.ndarray
    indices: np.ndarray
    num_threads: int
    streak_data: np.ndarray
    streak_mask: np.ndarray
    streaks: Dict[int, np.ndarray]

    def __init__(self, parent: ReferenceType, lsd_obj: LSD,
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
        init_funcs = {'lsd_obj': self.update_lsd.inplace_update}
        super(StreakDetector, self).__init__(init_funcs=init_funcs, parent=parent,
                                             lsd_obj=lsd_obj, **kwargs)

    @dict_to_object
    def update_lsd(self, scale: float=0.9, sigma_scale: float=0.9,
                   log_eps: float=0., ang_th: float=60.0, density_th: float=0.5,
                   quant: float=2.0e-2) -> StreakDetector:
        return {'lsd_obj': LSD(scale=scale, sigma_scale=sigma_scale, log_eps=log_eps,
                               ang_th=ang_th, density_th=density_th, quant=quant,
                               y_c=self.center[0], x_c=self.center[1])}

    @dict_to_object
    def update_mask(self, dilation: int=15, radius: float=1.0,
                    filter_lines: bool=True) -> StreakDetector:
        out_dict = self.lsd_obj.mask(self.streak_data, max_val=1, dilation=dilation,
                                     radius=radius, filter_lines=filter_lines,
                                     return_lines=True, num_threads=self.num_threads)
        return {'streak_mask': out_dict['mask'],
                'streaks': {self.indices[idx]: lines
                            for idx, lines in out_dict['lines'].items()}}

    @dict_to_object
    def update_streak_data(self, iterations: int=10) -> StreakDetector:
        if self.streak_mask is None:
            raise AttributeError("'streak_mask' must be generated before.")

        divisor = self.data
        for _ in range(iterations):
            divisor = maximum_filter(divisor, mask=self.streak_mask,
                                     footprint=self.footprint,
                                     num_threads=self.num_threads)
        streak_data = np.where(divisor, np.divide(self.data, divisor,
                                                  dtype=np.float64), 0.0)
        return {'streak_data': streak_data}
