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

class CXILoader(CXIProtocol):
    """CXI file loader class. Loads data from a CXI file and returns
    a :class:`CrystData` container or a :class:`dict` with the data. Search
    data in the paths provided by `protocol` and `load_paths`.

    Attributes:
        datatypes : Dictionary with attributes' datatypes. 'float', 'int',
            or 'bool' are allowed.
        default_paths : Dictionary with attributes' CXI default file paths.
        is_data : Dictionary with the flags if the attribute is of data
            type. Data type is 2- or 3-dimensional and has the same data
            shape as `data`.
        load_paths : Extra set of paths to the attributes enlisted in
            `datatypes`.
        policy: Loading policy. If a flag for the given attribute is
            True, the attribute will be loaded from a file.

    See Also:
        :class:`cbclib.CrystData` : Data container with all the data necessary
            for crystallography data processing.
    """
    attr_dict = {'datatypes': ('ALL', ),'default_paths': ('ALL', ),
                 'load_paths': ('ALL',), 'is_data': ('ALL', ), 'policy': ('ALL', )}
    fmt_dict = {'datatypes': 'str', 'default_paths': 'str',
                'load_paths': 'str', 'is_data': 'str', 'policy': 'str'}

    def __init__(self, protocol: CXIProtocol, load_paths: Dict[str, List[str]],
                 policy: Dict[str, Union[str, bool]]) -> None:
        """
        Args:
            protocol : Protocol object.
            load_paths : Extra paths to the data attributes in a CXI file,
                which override `protocol`. Accepts only the attributes
                enlisted in `protocol`.
            policy : A dictionary with loading policy. Contains all the
                attributes that are available in `protocol` with their
                corresponding flags. If a flag is True, the attribute
                will be loaded from a file.
        """
        load_paths = {attr: self.str_to_list(paths) for attr, paths in load_paths.items()
                      if attr in protocol}
        policy = {attr: flag for attr, flag in policy.items() if attr in protocol}
        super(CXIProtocol, self).__init__(datatypes=protocol.datatypes,
                                          default_paths=protocol.default_paths,
                                          is_data=protocol.is_data, load_paths=load_paths,
                                          policy=policy)

    @staticmethod
    def str_to_list(strings: Union[str, List[str]]) -> List[str]:
        """Convert `strings` to a list of strings.

        Args:
            strings : String or a list of strings

        Returns:
            List of strings.
        """
        if isinstance(strings, (str, list)):
            if isinstance(strings, str):
                return [strings,]
            return strings

        raise ValueError('strings must be a string or a list of strings')

    @classmethod
    def import_default(cls, protocol: Optional[CXIProtocol]=None,
                       load_paths: Optional[Dict[str, List[str]]]=None,
                       policy: Optional[Dict[str, Union[str, bool]]]=None) -> CXILoader:
        """Return the default :class:`CXILoader` object. Extra arguments
        override the default values if provided.

        Args:
            protocol : Protocol object.
            load_paths : Extra paths to the data attributes in a CXI file,
                which override `protocol`. Accepts only the attributes
                enlisted in `protocol`.
            policy : A dictionary with loading policy. Contains all the
                attributes that are available in `protocol` and the
                corresponding flags. If a flag is True, the attribute
                will be loaded from a file.

        Returns:
            A :class:`CXILoader` object with the default parameters.
        """
        return cls.import_ini(CXI_PROTOCOL, protocol, load_paths, policy)

    @classmethod
    def import_ini(cls, ini_file: str, protocol: Optional[CXIProtocol]=None,
                   load_paths: Optional[Dict[str, List[str]]]=None,
                   policy: Optional[Dict[str, Union[str, bool]]]=None) -> CXILoader:
        """Initialize a :class:`CXILoader` object class with an
        ini file.

        Args:
            ini_file : Path to the ini file. Loads the default CXI loader if None.
            protocol : Protocol object. Initialized with `ini_file` if None.
            load_paths : Extra paths to the data attributes in a CXI file,
                which override `protocol`. Accepts only the attributes
                enlisted in `protocol`. Initialized with `ini_file`
                if None.
            policy : A dictionary with loading policy. Contains all the
                attributes that are available in `protocol` and the
                corresponding flags. If a flag is True, the attribute
                will be loaded from a file. Initialized with `ini_file`
                if None.

        Returns:
            A :class:`CXILoader` object with all the attributes imported
            from the ini file.
        """
        if protocol is None:
            protocol = CXIProtocol.import_ini(ini_file)
        kwargs = cls._import_ini(ini_file)
        if not load_paths is None:
            kwargs['load_paths'].update(**load_paths)
        if not policy is None:
            kwargs['policy'].update(**policy)
        return cls(protocol=protocol, load_paths=kwargs['load_paths'],
                   policy=kwargs['policy'])

    def get_load_paths(self, attr: str, value: Optional[Union[str, List[str]]]=None) -> List[str]:
        """Return the atrribute's path in the cxi file.
        Return `value` if `attr` is not found.

        Args:
            attr : The attribute to look for.
            value : Value which is returned if the `attr` is not
                found.

        Returns:
            Set of attribute's paths.
        """
        paths = self.str_to_list(super(CXILoader, self).get_default_path(attr, value))
        if attr in self.load_paths:
            paths.extend(self.load_paths[attr])
        return paths

    def get_policy(self, attr: str, value: bool=False) -> bool:
        """Return the atrribute's loding policy.

        Args:
            attr : The attribute to look for.
            value : Value which is returned if the `attr` is not
                found.

        Returns:
            Attributes' loding policy.
        """
        policy = self.policy.get(attr, value)
        if isinstance(policy, str):
            return policy in ['True', 'true', '1', 'y', 'yes']
        else:
            return bool(policy)

    def get_protocol(self) -> CXIProtocol:
        """Return a CXI protocol from the loader.

        Returns:
            CXI protocol.
        """
        return CXIProtocol(datatypes=self.datatypes,
                           default_paths=self.default_paths,
                           is_data=self.is_data)

    def find_path(self, attr: str, cxi_file: h5py.File) -> str:
        """Find attribute's path in a CXI file `cxi_file`.

        Args:
            attr : Data attribute.
            cxi_file : :class:`h5py.File` object of the CXI file.

        Returns:
            Atrribute's path in the CXI file, returns an empty
            string if the attribute is not found.
        """
        paths = self.get_load_paths(attr)
        for path in paths:
            if path in cxi_file:
                return path
        return str()

    def load_attributes(self, master_file: str) -> Dict[str, Any]:
        """Return attributes' values from a CXI file at
        the given `master_file`.

        Args:
            master_file : Path to the master CXI file.

        Returns:
            Dictionary with the attributes retrieved from
            the CXI file.
        """
        attr_dict = {}
        with h5py.File(master_file, 'r') as cxi_file:
            for attr in self:
                cxi_path = self.find_path(attr, cxi_file)
                if not self.get_is_data(attr) and self.get_policy(attr, False) and cxi_path:
                    attr_dict[attr] = self.read_cxi(attr, cxi_file, cxi_path)
        return attr_dict

    def read_indices(self, attr: str, data_files: Union[str, List[str]]) -> np.ndarray:
        """Retrieve the indices of the datasets from the CXI files for the
        given attribute `attr`.

        Args:
            attr : The attribute to read.
            data_files : Paths to the data CXI files.

        Returns:
            A tuple of ('paths', 'cxi_paths', 'indices'). The elements
            are the following:

            * 'paths' : List of the file paths. None if no data is found.
            * 'cxi_paths' : List of the paths inside the files. None if no
              data is found.
            * 'indices' : List of the frame indices. None if no data is found.

        """
        data_files = self.str_to_list(data_files)

        paths, cxi_paths, indices = [], [], []
        for path in data_files:
            with h5py.File(path, 'r') as cxi_file:
                shapes = self.read_shape(cxi_file, self.find_path(attr, cxi_file))
            if shapes:
                for cxi_path, dset_shape in shapes:
                    if len(dset_shape) == 3:
                        paths.extend(np.repeat(path, dset_shape[0]).tolist())
                        cxi_paths.extend(np.repeat(cxi_path, dset_shape[0]).tolist())
                        indices.extend(np.arange(dset_shape[0]).tolist())
                    elif len(dset_shape) == 2:
                        paths.append(path)
                        cxi_paths.append(cxi_path)
                        indices.append(slice(None))
                    else:
                        raise ValueError(f'{attr:s} dataset must be 2- or 3-dimensional: {str(dset_shape):s}')

        return np.array([paths, cxi_paths, indices], dtype=object).T

    @staticmethod
    def _read_frame(index):
        with h5py.File(index[0], 'r') as cxi_file:
            return cxi_file[index[1]][index[2]]

    def load_data(self, attr: str, indices: np.ndarray, verbose: bool=True, processes: int=1) -> np.ndarray:
        """Retrieve the data for the given attribute `attr` from the
        CXI files. Uses the result from :func:`CXILoader.read_indices`
        method.

        Args:
            attr : The attribute to read.
            paths : List of the file paths.
            cxi_paths : List of the paths inside the files.
            indices : List of the frame indices.
            verbose : Print the progress bar if True.

        Returns:
            Data array retrieved from the CXI files.
        """

        data = []
        with Pool(processes=processes) as pool:
            for frame in tqdm(pool.imap(CXILoader._read_frame, indices), total=indices.shape[0],
                              desc=f'Loading {attr:s}', disable=not verbose):
                data.append(frame)

        if len(data) == 1:
            data = data[0]
        else:
            data = np.stack(data, axis=0)
        return np.asarray(data, dtype=self.get_dtype(attr))

    def load_to_dict(self, data_files: Union[str, List[str]],
                     master_file: Optional[str]=None,
                     frame_indices: Optional[Iterable[int]]=None, processes: int=1,
                     **attributes: Any) -> Dict[str, Any]:
        """Load data from the CXI files and return a :class:`dict` with
        all the data fetched from the `data_files` and `master_file`.

        Args:
            data_files : Paths to the data CXI files.
            master_file : Path to the master CXI file. First file in `data_files`
                if not provided.
            frame_indices : Array of frame indices to load. Loads all the frames
                by default.
            attributes : Dictionary of attribute values, that override the loaded
                values.

        Returns:
            Dictionary with all the data fetched from the CXI files.
        """
        if master_file is None:
            if isinstance(data_files, str):
                master_file = data_files
            elif isinstance(data_files, list):
                master_file = data_files[0]
            else:
                raise ValueError('data_files must be a string or a list of strings')

        data_dict = self.load_attributes(master_file)

        if frame_indices is None:
            n_frames = 0
            for attr in self:
                if self.get_is_data(attr) and self.get_policy(attr):
                    n_frames = max(self.read_indices(attr, data_files).shape[0], n_frames)
            frame_indices = np.arange(n_frames)
        else:
            frame_indices = np.asarray(frame_indices)

        if frame_indices.size:
            for attr in self:
                if self.get_is_data(attr) and self.get_policy(attr):
                    indices = self.read_indices(attr, data_files)
                    if indices.shape[0] > 0:
                        good_frames = frame_indices[frame_indices < indices.shape[0]]
                        data_dict[attr] = self.load_data(attr, indices[good_frames],
                                                         processes=processes)

        if 'frames' not in data_dict:
            data_dict['frames'] = frame_indices
        else:
            data_dict['frames'] = data_dict['frames'][frame_indices]

        for attr, val in attributes.items():
            if attr in self and val is not None:
                if isinstance(val, dict):
                    data_dict[attr] = {dkey: self.get_dtype(attr)(dval)
                                       for dkey, dval in val.items()}
                else:
                    data_dict[attr] = np.asarray(val, dtype=self.get_dtype(attr))
                    if data_dict[attr].size == 1:
                        data_dict[attr] = data_dict[attr].item()

        return data_dict

    def load(self, data_files: str, master_file: Optional[str]=None,
             frame_indices: Optional[Iterable[str]]=None, processes: int=1,
             **attributes: Any) -> CrystData:
        """Load data from the CXI files and return a :class:`STData` container
        with all the data fetched from the `data_files` and `master_file`.

        Args:
            data_files : Paths to the data CXI files.
            master_file : Path to the master CXI file. First file in `data_files`
                if not provided.
            frame_indices : Array of frame indices to load. Loads all the frames by
                default.
            attributes : Dictionary of attribute values, which will be parsed to
                the :class:`cbclib.Crystdata` object instead.

        Returns:
            Data container object with all the necessary data for crystallography
            data processing.
        """
        return CrystData(self.get_protocol(), **self.load_to_dict(data_files, master_file,
                                                                  frame_indices, processes,
                                                                  **attributes))

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
