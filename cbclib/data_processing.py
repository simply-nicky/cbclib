from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from multiprocessing import cpu_count, Pool
from tqdm.auto import tqdm
import h5py
import numpy as np
from weakref import ref, ReferenceType
from .cxi_protocol import CXI_PROTOCOL, CXIProtocol
from .data_container import DataContainer, dict_to_object
from .bin import subtract_background, median, median_filter, LSD, maximum_filter

class CXIStore():

    # _index_converters = {'stack': self._stack_shapes_to_indices,
    #                      'frame': self._frame_shapes_to_indices,
    #                      'sequence': self._sequence_shapes_to_indices,
    #                      'scalar': self._scalar_shapes_to_indices}

    def __init__(self, data_files, protocol):
        self.protocol = protocol

        self.frame_shape = None

    def indices(self):
        pass

    def _stack_shapes_to_indices(self, path, shapes):
        paths, cxi_paths, fidxs = [], [], []

        for cxi_path, shape in shapes.items():
            if len(shape) != 3 or shape[1:] != self.frame_shape:
                err_txt = f'Dataset at {path}: {cxi_path:s} has invalid shape: {str(shape):s}'
                raise ValueError(err_txt)

            paths.extend(np.repeat(path, shape[0]).tolist())
            cxi_paths.extend(np.repeat(cxi_path, shape[0]).tolist())
            fidxs.extend(np.arange(shape[0]).tolist())

        indices = np.array([paths, cxi_paths, fidxs], dtype=object).T
        return indices[self.indices()]

    def _frame_shapes_to_indices(self, path, shapes):
        paths, cxi_paths, fidxs = [], [], []

        for cxi_path, shape in shapes.items():
            if shape != self.frame_shape:
                err_txt = f'Dataset at {path}: {cxi_path:s} has invalid shape: {str(shape):s}'
                raise ValueError(err_txt)

            paths.append(path)
            cxi_paths.append(cxi_path)
            fidxs.append(tuple())

        indices = np.array([paths, cxi_paths, fidxs], dtype=object).T
        return indices[self.indices()]

    def _sequence_shapes_to_indices(self, path, shapes):
        paths, cxi_paths, fidxs = [], [], []

        for cxi_path, shape in shapes.items():
            if len(shape) != 1:
                err_txt = f'Dataset at {path}: {cxi_path:s} has invalid shape: {str(shape):s}'
                raise ValueError(err_txt)

            paths.append(path)
            cxi_paths.append(cxi_path)
            fidxs.append(tuple())

        indices = np.array([paths, cxi_paths, fidxs], dtype=object).T
        return indices[self.indices()]

    def _scalar_shapes_to_indices(self, path, shapes):
        paths, cxi_paths, fidxs = [], [], []

        for cxi_path, shape in shapes.items():
            if len(shape) <= 1:
                err_txt = f'Dataset at {path}: {cxi_path:s} has invalid shape: {str(shape):s}'
                raise ValueError(err_txt)

            paths.append(path)
            cxi_paths.append(cxi_path)
            fidxs.append(tuple())

        return np.array([paths, cxi_paths, fidxs], dtype=object).T

class CrystData(DataContainer):
    attr_set = {'protocol', 'frames', 'data'}
    init_set = {'cor_data', 'flatfields', 'good_frames', 'mask', 'num_threads',
                'pupil', 'roi', 'streak_data', 'streaks', 'tilts', 'translations',
                'whitefield'}

    inits = {'num_threads': lambda obj: np.clip(1, 64, cpu_count()),
             'roi'        : lambda obj: np.array([0, obj.data.shape[1], 0, obj.data.shape[2]]),
             'good_frames': lambda obj: np.arange(obj.data.shape[0]),
             'mask'       : lambda obj: obj._generate_mask(),
             'cor_data'   : lambda obj: obj.update_cor_data.inplace_update()}

    def __init__(self, protocol=CXIProtocol.import_default(), **kwargs):
        super(CrystData, self).__init__(protocol=protocol, **kwargs)

        if self.tilts is not None and len(self.tilts) != self.frames.size:
            self.tilts = {frame: self.tilts[frame] for frame in self.frames}
        if self.translations is not None and len(self.translations) != self.frames.size:
            self.translations = {frame: self.translations[frame] for frame in self.frames}
        if self.streaks is not None and len(self.streaks) != self.frames.size:
            self.streaks = {frame: self.streaks[frame] for frame in self.frames}

    def _generate_mask(self) -> np.ndarray:
        mask = np.ones(self.data.shape, dtype=self.protocol.get_dtype('mask'))
        if self.pupil is not None:
            mask[:, self.pupil[0]:self.pupil[1], self.pupil[2]:self.pupil[3]] = False
        return mask

    @property
    def _iswhitefield(self):
        return not self.whitefield is None

    def __setattr__(self, attr, value):
        if attr in self and 'protocol' in self.__dict__:
            dtype = self.protocol.get_dtype(attr)
            if isinstance(value, dict):
                for key in value:
                    if isinstance(value[key], np.ndarray):
                        value[key] = np.asarray(value[key], dtype=dtype)
            if isinstance(value, np.ndarray):
                value = np.asarray(value, dtype=dtype)
            super(CrystData, self).__setattr__(attr, value)
        else:
            super(CrystData, self).__setattr__(attr, value)

    @dict_to_object
    def bin_data(self, bin_ratio: int=2) -> CrystData:
        """Return a new :class:`CrystData` object with the data binned by
        a factor `bin_ratio`.

        Args:
            bin_ratio : Binning ratio. The frame size will decrease by
                the factor of `bin_ratio`.

        Returns:
            New :class:`CrystData` object with binned `data`.
        """
        data_dict = {'data': self.data[:, ::bin_ratio, ::bin_ratio],
                     'mask': self.mask[:, ::bin_ratio, ::bin_ratio],
                     'roi': self.roi // bin_ratio + (self.roi % bin_ratio > 0)}
        if self.pupil is not None:
            data_dict['pupil'] = self.pupil // bin_ratio

        if self._iswhitefield:
            data_dict['whitefield'] = self.whitefield[::bin_ratio, ::bin_ratio]
            data_dict['cor_data'] = self.cor_data[:, ::bin_ratio, ::bin_ratio]

        if self.streak_data is not None:
            data_dict['streaks'] = {frame: lines / bin_ratio
                                    for frame, lines in self.streaks.items()}

        if self.streak_data is not None:
            data_dict['streak_data'] = self.streak_data[:, ::bin_ratio, ::bin_ratio]

        return data_dict

    @dict_to_object
    def crop_data(self, roi: Iterable[int]) -> CrystData:
        """Return a new :class:`CrystData` object with the updated `roi`.

        Args:
            roi : Region of interest in the detector plane.

        Returns:
            New :class:`CrystData` object with the updated `roi`.
        """
        return {'roi': np.asarray(roi, dtype=int)}

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
        return {'good_frames': np.asarray(good_frames, dtype=np.int)}

    def get(self, attr: str, value: Optional[Any]=None) -> Any:
        """Return a dataset with `mask` and `roi` applied.
        Return `value` if the attribute is not found.

        Args:
            attr : Attribute to fetch.
            value : Return if `attr` is not found.

        Returns:
            `attr` dataset with `mask` and `roi` applied.
            `value` if `attr` is not found.
        """
        if attr in self:
            val = super(CrystData, self).get(attr)
            if val is not None:
                if self.protocol.get_is_data(attr):
                    val = val[..., self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
                if attr in ['frames', 'data', 'cor_data', 'flatfields', 'mask',
                            'streak_data', 'translations']:
                    val = val[self.good_frames]
            return val
        return value

    def get_pca(self) -> Tuple[np.ndarray, np.ndarray]:
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
        return eig_vals / eig_vals.sum(), effs

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
    def update_cor_data(self):
        if self._iswhitefield:
            cor_data = np.zeros(self.data.shape, dtype=self.protocol.get_dtype('cor_data'))
            good_cdata = subtract_background(data=self.get('data'), mask=self.get('mask'),
                                             whitefield=self.get('whitefield'),
                                             flatfields=self.get('flatfields'),
                                             num_threads=self.num_threads)
            cor_data[self.good_frames, self.roi[0]:self.roi[1],
                     self.roi[2]:self.roi[3]] = good_cdata
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
            good_wf = self.get('whitefield')

            if method == 'median':
                good_data = self.get('data')
                outliers = np.abs(good_data - good_wf) < 3 * np.sqrt(good_wf)
                good_flats = median_filter(good_data, size=(size, 1, 1), mask=outliers,
                                        num_threads=self.num_threads)
            elif method == 'pca':
                if effs is None:
                    raise ValueError('No eigen flat fields were provided')

                weights = np.tensordot(self.cor_data, effs, axes=((1, 2), (1, 2))) / \
                          np.sum(effs * effs, axis=(1, 2))
                good_flats = np.tensordot(weights, effs, axes=((1,), (0,))) + good_wf
            else:
                raise ValueError('Invalid method argument')

            flatfields = np.zeros(self.data.shape, dtype=self.protocol.get_dtype('flatfields'))
            flatfields[self.good_frames, self.roi[0]:self.roi[1],
                       self.roi[2]:self.roi[3]] = good_flats
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
            data = self.get('data')
        elif update == 'multiply':
            data = self.get('data') * self.get('mask')
        else:
            raise ValueError(f'Invalid update keyword: {update:s}')

        if method == 'no-bad':
            mask = np.ones((self.good_frames.size, self.roi[1] - self.roi[0],
                            self.roi[3] - self.roi[2]), dtype=bool)
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
        mask_full = self.mask.copy()

        if self.pupil is not None:
            mask_full[:, self.pupil[0]:self.pupil[1], self.pupil[2]:self.pupil[3]] = False

        if update == 'reset':
            mask_full[self.good_frames, self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]] = mask
        elif update == 'multiply':
            mask_full[self.good_frames, self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]] *= mask
        else:
            raise ValueError(f'Invalid update keyword: {update:s}')

        return {'mask': mask_full, 'cor_data': None}

    @dict_to_object
    def update_whitefield(self, method='median'):
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
            good_wf = median(data=self.get('data'), mask=self.get('mask'), axis=0,
                             num_threads=self.num_threads)
        elif method == 'mean':
            good_wf = np.mean(self.get('data') * self.get('mask'), axis=0)
        else:
            raise ValueError('Invalid method argument')

        whitefield = np.zeros(self.data.shape[1:], dtype=self.protocol.get_dtype('whitefield'))
        whitefield[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]] = good_wf
        return {'whitefield': whitefield, 'cor_data': None}

    def detect_streaks(self, vmin, vmax, size=(1, 3, 3)):
        if self.pupil is None or not self._iswhitefield:
            raise ValueError('No pupil and whitefield in the container')

        data = self.get('cor_data')
        if size is not None:
            data = median_filter(data, size=size, num_threads=self.num_threads)

        streak_data = np.divide(np.clip(data, vmin, vmax) - vmin, vmax - vmin, dtype=np.float64)
        center = np.array([self.pupil[:2].mean() - self.roi[0],
                           self.pupil[2:].mean() - self.roi[2]])
        return StreakDetector(parent=ref(self), center=center, indices=self.get('frames'),
                              data=data, streak_data=streak_data,num_threads=self.num_threads)

    def update_streaks(self, det_obj):
        if det_obj.parent() is not self:
            raise ValueError("The StreakDetector object doesn't belong to the data container")

        self.streak_data = np.zeros(self.data.shape, dtype=self.protocol.get_dtype('streaks'))
        self.streak_data[self.good_frames[det_obj.indices], self.roi[0]:self.roi[1],
                         self.roi[2]:self.roi[3]] = det_obj.streak_data

        self.streaks = {}
        for frame in det_obj.streaks:
            self.streaks[frame] = det_obj.streaks[frame].copy()
            self.streaks[frame][:, :4:2] += self.roi[2]
            self.streaks[frame][:, 1:4:2] += self.roi[0]

    def write_cxi(self, cxi_file: h5py.File) -> None:
        """Write all the `attr` to a CXI file `cxi_file`.

        Args:
            cxi_file : :class:`h5py.File` object of the CXI file.
            overwrite : Overwrite the content of `cxi_file` file if it's True.

        Raises:
            ValueError : If `overwrite` is False and the data is already present
                in `cxi_file`.
        """
        old_frames = self.protocol.read_cxi('frames', cxi_file)

        if old_frames is None:
            for attr, data in self.items():
                if attr in self.protocol:
                    self.protocol.write_cxi(attr, data, cxi_file)

        else:
            frames, idxs = np.unique(np.concatenate((old_frames, self.frames)),
                                     return_inverse=True)
            old_idxs, new_idxs = idxs[:old_frames.size], idxs[old_frames.size:]

            for attr, data in self.items():
                if attr in self.protocol and data is not None:
                    if attr in ['streaks', 'tilts', 'translations']:
                        old_data = self.protocol.read_cxi(attr, cxi_file)
                        if old_data is not None:
                            data.update(old_data)

                        self.protocol.write_cxi(attr, data, cxi_file)

                    elif self.protocol.get_is_data(attr):
                        dset = cxi_file[self.protocol.get_default_path(attr)]
                        dset.resize((frames.size,) + self.data.shape[1:])

                        if np.all(old_idxs != np.arange(old_frames.size)):
                            old_data = self.protocol.read_cxi(attr, cxi_file)
                            if old_data is not None:
                                dset[old_idxs] = old_data

                        dset[new_idxs] = data

                    elif attr == 'frames':
                        self.protocol.write_cxi(attr, frames, cxi_file)

                    else:
                        self.protocol.write_cxi(attr, data, cxi_file)

class StreakDetector(DataContainer):
    attr_set = {'parent', 'center', 'data', 'indices', 'num_threads', 'streak_data'}
    init_set = {'lsd_obj', 'streak_mask', 'streaks'}
    footprint = np.array([[[False, False,  True, False, False],
                           [False,  True,  True,  True, False],
                           [ True,  True,  True,  True,  True],
                           [False,  True,  True,  True, False],
                           [False, False,  True, False, False]]])

    inits = {'lsd_obj': lambda obj: obj.update_lsd.inplace_update()}

    def __init__(self, parent: ReferenceType, **kwargs: Union[int, float, np.ndarray]) -> None:
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
        super(StreakDetector, self).__init__(parent=parent, **kwargs)

    @dict_to_object
    def update_lsd(self, scale=0.9, sigma_scale=0.9, log_eps=0.,
                   ang_th=60.0, density_th=0.5, quant=2.0e-2):
        return {'lsd_obj': LSD(scale=scale, sigma_scale=sigma_scale, log_eps=log_eps,
                               ang_th=ang_th, density_th=density_th, quant=quant,
                               y_c=self.center[0], x_c=self.center[1])}

    @dict_to_object
    def update_mask(self, dilation=15, radius=1.0, filter_lines=True):
        out_dict = self.lsd_obj.mask(self.streak_data, max_val=1, dilation=dilation,
                                     radius=radius, filter_lines=filter_lines,
                                     return_lines=True, num_threads=self.num_threads)
        return {'streak_mask': out_dict['mask'],
                'streaks': {self.indices[idx]: lines for idx, lines in out_dict['lines'].items()}}

    @dict_to_object
    def update_streak_data(self, iterations=10):
        if self.streak_mask is None:
            raise AttributeError("'streak_mask' must be generated before.")

        divisor = self.data.copy()
        for _ in range(iterations):
            divisor = maximum_filter(divisor, mask=self.streak_mask,
                                     footprint=self.footprint,
                                     num_threads=self.num_threads)
        streak_data = np.where(divisor, np.divide(self.data, divisor, dtype=np.float64), 0.0)
        return {'streak_data': streak_data}
