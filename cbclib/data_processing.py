from __future__ import annotations
from multiprocessing import cpu_count
from typing import (Any, Dict, ItemsView, Iterable, Iterator, KeysView, List, Optional, Set, Tuple, Union,
                    ValuesView)
from dataclasses import dataclass, field
from weakref import ref, ReferenceType
import numpy as np
import pandas as pd
from .cxi_protocol import CXIStore, Indices
from .data_container import DataContainer, dict_to_object
from .ini_parser import INIParser
from .bin import (tilt_matrix, subtract_background, project_effs, median, median_filter, LSD,
                  maximum_filter, normalize_streak_data, draw_lines, draw_lines_stack,
                  draw_line_indices)

FloatArray = Union[List[float], Tuple[float, ...], np.ndarray]

class ScanSetup(INIParser):
    """
    Detector tilt scan experimental setup class

    foc_pos - focus position relative to the detector [m]
    pix_size - detector pixel size [m]
    rot_axis - axis of rotation
    smp_pos - sample position relative to the detector [m]
    """
    attr_dict = {'exp_geom': ('foc_pos', 'rot_axis', 'wavelength', 'x_pixel_size',
                              'y_pixel_size', 'kin_min', 'kin_max')}
    fmt_dict = {'exp_geom': 'float'}

    foc_pos         : np.ndarray
    rot_axis        : np.ndarray
    kin_min         : np.ndarray
    kin_max         : np.ndarray
    wavelength      : float
    x_pixel_size    : float
    y_pixel_size    : float

    def __init__(self, foc_pos: FloatArray, rot_axis: FloatArray,
                 kin_min: FloatArray, kin_max: FloatArray, wavelength: float,
                 x_pixel_size: float, y_pixel_size: float) -> None:
        exp_geom = {'foc_pos': foc_pos, 'rot_axis': rot_axis, 'kin_min': kin_min,
                    'kin_max': kin_max, 'wavelength': wavelength,
                    'x_pixel_size': x_pixel_size, 'y_pixel_size': y_pixel_size}
        super(ScanSetup, self).__init__(exp_geom=exp_geom)

    @classmethod
    def _lookup_dict(cls) -> Dict[str, str]:
        lookup = {}
        for section in cls.attr_dict:
            for option in cls.attr_dict[section]:
                lookup[option] = section
        return lookup

    @property
    def kin_center(self) -> np.ndarray:
        return 0.5 * (self.kin_max + self.kin_min)

    def __iter__(self) -> Iterator[str]:
        return self._lookup.__iter__()

    def __contains__(self, attr: str) -> bool:
        return attr in self._lookup

    def __repr__(self) -> str:
        return self._format(self.export_dict()).__repr__()

    def __str__(self) -> str:
        return self._format(self.export_dict()).__str__()

    def keys(self) -> KeysView[str]:
        return self._lookup.keys()

    @classmethod
    def import_ini(cls, ini_file: str, **kwargs: Any) -> ScanSetup:
        """Initialize a :class:`ScanSetup` object with an
        ini file.

        Parameters
        ----------
        ini_file : str, optional
            Path to the ini file. Load the default parameters
            if None.
        **kwargs : dict
            Experimental geometry parameters.
            Initialized with `ini_file` if not provided.

        Returns
        -------
        scan_setup : ScanSetup
            A :class:`ScanSetup` object with all the attributes
            imported from the ini file.
        """
        attr_dict = cls._import_ini(ini_file)
        for option, section in cls._lookup_dict().items():
            if option in kwargs:
                attr_dict[section][option] = kwargs[option]
        return cls(**attr_dict['exp_geom'])

    def _det_to_k(self, x: np.ndarray, y: np.ndarray, source: np.ndarray) -> np.ndarray:
        delta_y = y * self.y_pixel_size - source[1]
        delta_x = x * self.x_pixel_size - source[0]
        phis = np.arctan2(delta_y, delta_x)
        thetas = np.arctan(np.sqrt(delta_x**2 + delta_y**2) / source[2])
        return np.stack((np.sin(thetas) * np.cos(phis),
                         np.sin(thetas) * np.sin(phis),
                         np.cos(thetas)), axis=-1)

    def _k_to_det(self, karr: np.ndarray, source: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        theta = np.arccos(karr[..., 2] / np.sqrt((karr * karr).sum(axis=-1)))
        phi = np.arctan2(karr[..., 1], karr[..., 0])
        det_x = source[2] * np.tan(theta) * np.cos(phi) + source[0]
        det_y = source[2] * np.tan(theta) * np.sin(phi) + source[1]
        return det_x / self.x_pixel_size, det_y / self.y_pixel_size

    def detector_to_kout(self, x: np.ndarray, y: np.ndarray, smp_pos: np.ndarray) -> np.ndarray:
        return self._det_to_k(x, y, smp_pos)

    def kout_to_detector(self, kout: np.ndarray, smp_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self._k_to_det(kout, smp_pos)

    def detector_to_kin(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self._det_to_k(x, y, self.foc_pos)

    def kin_to_detector(self, kin: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self._k_to_det(kin, self.foc_pos)

    def tilt_matrices(self, tilts: Union[float, np.ndarray]) -> np.ndarray:
        return tilt_matrix(np.atleast_1d(tilts), self.rot_axis)

class Transform():
    """Abstract transform class."""

    def index_array(self, ss_idxs: np.ndarray, fs_idxs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.state_dict().__repr__()

    def __str__(self) -> str:
        return self.state_dict().__str__()

    def forward(self, inp: np.ndarray) -> np.ndarray:
        ss_idxs, fs_idxs = np.indices(inp.shape[-2:])
        ss_idxs, fs_idxs = self.index_array(ss_idxs, fs_idxs)
        return inp[..., ss_idxs, fs_idxs]

    def forward_points(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def backward(self, inp: np.ndarray, out: np.ndarray) -> np.ndarray:
        ss_idxs, fs_idxs = np.indices(out.shape[-2:])
        ss_idxs, fs_idxs = self.index_array(ss_idxs, fs_idxs)
        out[..., ss_idxs, fs_idxs] = inp
        return out

    def backward_points(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def state_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

class Crop(Transform):
    """Crop transform. Crops a frame according to a region of interest.

    Attributes:
        roi : Region of interest. Comprised of four elements `[y_min, y_max,
            x_min, x_max]`.
    """
    def __init__(self, roi: Union[List[int], Tuple[int, int, int, int], np.ndarray]) -> None:
        """
        Args:
            roi : Region of interest. Comprised of four elements `[y_min, y_max,
                x_min, x_max]`.
        """
        self.roi = roi

    def __eq__(self, obj: object) -> bool:
        if isinstance(obj, Crop):
            return self.roi[0] == obj.roi[0] and self.roi[1] == obj.roi[1] and \
                   self.roi[2] == obj.roi[2] and self.roi[3] == obj.roi[3]
        return NotImplemented

    def __ne__(self, obj: object) -> bool:
        if isinstance(obj, Crop):
            return self.roi[0] != obj.roi[0] or self.roi[1] != obj.roi[1] or \
                   self.roi[2] != obj.roi[2] or self.roi[3] != obj.roi[3]
        return NotImplemented

    def index_array(self, ss_idxs: np.ndarray, fs_idxs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return (ss_idxs[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]],
                fs_idxs[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]])

    def forward_points(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply the transform to a set of points.

        Args:
            pts : Input array of points.

        Returns:
            Output array of points.
        """
        return x - self.roi[2], y - self.roi[0]

    def backward_points(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Tranform back an array of points.

        Args:
            pts : Input tranformed array of points.

        Returns:
            Output array of points.
        """
        return x + self.roi[2], y + self.roi[0]

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the transform as a dict.

        Returns:
            A dictionary with all the attributes.
        """
        return {'roi': self.roi[:]}

class Downscale(Transform):
    """Downscale the image by a integer ratio.

    Attributes:
        scale : Downscaling integer ratio.
    """
    def __init__(self, scale: int) -> None:
        """
        Args:
            scale : Downscaling integer ratio.
        """
        self.scale = scale

    def index_array(self, ss_idxs: np.ndarray, fs_idxs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return (ss_idxs[::self.scale, ::self.scale], fs_idxs[::self.scale, ::self.scale])

    def forward_points(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply the transform to a set of points.

        Args:
            pts : Input array of points.

        Returns:
            Output array of points.
        """
        return x / self.scale, y / self.scale

    def backward_points(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Tranform back an array of points.

        Args:
            pts : Input tranformed array of points.
            shape : Detector shape.

        Returns:
            Output array of points.
        """
        return x * self.scale, y * self.scale

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the transform as a dict.

        Returns:
            A dictionary with all the attributes.
        """
        return {'scale': self.scale}

class Mirror(Transform):
    """Mirror the data around an axis.

    Attributes:
        axis : Axis of reflection.
    """
    def __init__(self, axis: int, shape: Tuple[int, int]) -> None:
        """
        Args:
            axis : Axis of reflection.
        """
        if axis not in [0, 1]:
            raise ValueError('Axis must equal to 0 or 1')
        self.axis, self.shape = axis, shape

    def index_array(self, ss_idxs: np.ndarray, fs_idxs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.axis == 0:
            return (ss_idxs[::-1], fs_idxs[::-1])
        if self.axis == 1:
            return (ss_idxs[:, ::-1], fs_idxs[:, ::-1])
        raise ValueError('Axis must equal to 0 or 1')

    def forward_points(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply the transform to a set of points.

        Args:
            pts : Input array of points.

        Returns:
            Output array of points.
        """
        if self.axis:
            return x, self.shape[0] - y
        return self.shape[1] - x, y

    def backward_points(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Tranform back an array of points.

        Args:
            pts : Input tranformed array of points.

        Returns:
            Output array of points.
        """
        return self.forward_points(x, y)

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
    """
    transforms : List[Transform]

    def __init__(self, transforms: List[Transform]) -> None:
        """
        Args:
            transforms: List of transforms.
        """
        if len(transforms) < 2:
            raise ValueError('Two or more transforms are needed to compose')

        self.transforms = []
        for transform in transforms:
            pdict = transform.state_dict()
            self.transforms.append(type(transform)(**pdict))

    def __iter__(self) -> Iterator[Transform]:
        return self.transforms.__iter__()

    def __getitem__(self, idx: Union[int, slice]) -> Union[Transform, List[Transform]]:
        return self.transforms[idx]

    def index_array(self, ss_idxs: np.ndarray, fs_idxs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        for transform in self:
            ss_idxs, fs_idxs = transform.index_array(ss_idxs, fs_idxs)
        return ss_idxs, fs_idxs

    def forward_points(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply the transform to a set of points.

        Args:
            pts : Input array of points.

        Returns:
            Output array of points.
        """
        for transform in self:
            x, y = transform.forward_points(x, y)
        return x, y

    def backward_points(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Tranform back an array of points.

        Args:
            pts : Input tranformed array of points.

        Returns:
            Output array of points.
        """
        for transform in list(self)[::-1]:
            x, y = transform.backward_points(x, y)
        return x, y

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the transform as a dict.

        Returns:
            A dictionary with all the attributes.
        """
        return {'transforms': self.transforms[:]}

class CrystData(DataContainer):
    attr_set: Set[str] = {'input_file'}
    init_set: Set[str] = {'background', 'cor_data', 'data', 'good_frames', 'frames', 'mask',
                          'num_threads', 'output_file', 'streak_data', 'tilts', 'transform',
                          'translations', 'whitefield'}

    # Necessary attributes
    input_file:     CXIStore
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

    def __init__(self, input_file: CXIStore, output_file: Optional[CXIStore]=None,
                 transform: Optional[Transform]=None, **kwargs):
        super(CrystData, self).__init__(input_file=input_file, output_file=output_file,
                                        transform=transform, **kwargs)

        self._init_functions(num_threads=lambda: np.clip(1, 64, cpu_count()))
        if self.shape[0] > 0:
            self._init_functions(good_frames=lambda: np.arange(self.shape[0]))
        if self._isdata:
            self._init_functions(mask=lambda: np.ones(self.shape, dtype=bool))
            if self._iswhitefield:
                bgd_func = lambda: project_effs(self.data, mask=self.mask,
                                                effs=self.whitefield[None, ...],
                                                num_threads=self.num_threads)
                cor_func = lambda: subtract_background(self.data, mask=self.mask,
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

    def forward_points(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.transform:
            return self.transform.forward_points(x, y)
        return x, y

    def backward_points(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.transform:
            return self.transform.backward_points(x, y)
        return x, y

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
        with self.input_file:
            self.input_file.update_indices()
            shape = self.input_file.read_shape()

            if attributes is None:
                attributes = [attr for attr in self.input_file.keys()
                              if attr in self.init_set]
            else:
                attributes = self.input_file.protocol.str_to_list(attributes)

            if idxs is None:
                idxs = self.input_file.indices()
            data_dict = {'frames': idxs}

            for attr in attributes:
                if attr not in self.input_file.keys():
                    raise ValueError(f"No '{attr}' attribute in the input files")
                if attr not in self.init_set:
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

        data_dict: Dict[str, Any] = {}
        for attr in self.input_file.protocol.str_to_list(attributes):
            data = self.get(attr)
            if attr in self and isinstance(data, np.ndarray):
                data_dict[attr] = None
        return data_dict

    @dict_to_object
    def update_output_file(self, output_file: CXIStore) -> CrystData:
        """Return a new :class:`CrystData` object with the new output
        file handler.

        Args:
            output_file : A new output file handler.

        Returns:
            New :class:`CrystData` object with the new output file
            handler.
        """
        return {'output_file': output_file}

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
            if not self._isdata:
                raise ValueError('No data in the container')

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
        """Return a new :class:`CrystData` object with the updated mask. The region
        defined by the `[y_min, y_max, x_min, x_max]` will be masked out.

        Args:
            roi : Bad region of interest in the detector plane. A set of four
            coordinates `[y_min, y_max, x_min, x_max]`.

        Returns:
            New :class:`CrystData` object with the updated `mask`.
        """
        if not self._isdata:
            raise ValueError('No data in the container')

        mask = self.mask.copy()
        mask[:, roi[0]:roi[1], roi[2]:roi[3]] = False

        if self._iswhitefield:
            cor_data = self.cor_data.copy()
            cor_data[:, roi[0]:roi[1], roi[2]:roi[3]] = 0.0
            return {'cor_data': cor_data, 'mask': mask}

        return {'mask': mask}

    def mask_pupil(self, setup: ScanSetup, padding: float=0.0) -> CrystData:
        x0, y0 = self.forward_points(*setup.kin_to_detector(np.asarray(setup.kin_min)))
        x1, y1 = self.forward_points(*setup.kin_to_detector(np.asarray(setup.kin_max)))
        return self.mask_region((int(y0 - padding), int(y1 + padding),
                                 int(x0 - padding), int(x1 + padding)))

    @dict_to_object
    def blur_pupil(self, setup: ScanSetup, padding: float=0.0, blur: float=0.0) -> CrystData:
        if not self._iswhitefield:
            raise AttributeError("No whitefield in the container")

        x0, y0 = self.forward_points(*setup.kin_to_detector(np.asarray(setup.kin_min)))
        x1, y1 = self.forward_points(*setup.kin_to_detector(np.asarray(setup.kin_max)))

        i, j = np.indices(self.shape[1:])
        dtype = self.cor_data.dtype
        window = 0.25 * (np.tanh((i - y0 + padding) / blur, dtype=dtype) + \
                         np.tanh((y1 + padding - i) / blur, dtype=dtype)) * \
                        (np.tanh((j - x0 + padding) / blur, dtype=dtype) + \
                         np.tanh((x1 + padding - j) / blur, dtype=dtype))
        return {'cor_data': self.cor_data * (1.0 - window)}

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
        if sum(self.shape[1:]) and whitefield.shape != self.shape[1:]:
            raise ValueError('whitefield and data have incompatible shapes: '\
                             f'{whitefield.shape} != {self.shape[1:]}')
        return {'whitefield': whitefield, 'cor_data': None}

    @dict_to_object
    def update_cor_data(self) -> CrystData:
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
        if not self._isdata:
            raise ValueError('No data in the container')

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

        if self.transform is None:
            for attr, data in self.items():
                if attr in self.input_file.protocol and data is not None:
                    kind = self.input_file.protocol.get_kind(attr)
                    if kind in ['stack', 'frame']:
                        data = transform.forward(data)
                    data_dict[attr] = data

            return data_dict

        for attr, data in self.items():
            if attr in self.input_file.protocol and data is not None:
                kind = self.input_file.protocol.get_kind(attr)
                if kind in ['stack', 'frame']:
                    data_dict[attr] = None
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
        if not self._isdata:
            raise ValueError('No data in the container')

        if method == 'median':
            whitefield = median(self.data[self.good_frames], mask=self.mask[self.good_frames],
                                axis=0, num_threads=self.num_threads)
        elif method == 'mean':
            whitefield = np.mean(self.data[self.good_frames] * self.mask[self.good_frames], axis=0)
        else:
            raise ValueError('Invalid method argument')

        return {'whitefield': whitefield, 'cor_data': None}

    def get_detector(self, import_contents: bool=False) -> StreakDetector:
        if not self._iswhitefield:
            raise ValueError("No whitefield in the container")

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

@dataclass
class Streaks():
    x0          : np.ndarray
    y0          : np.ndarray
    x1          : np.ndarray
    y1          : np.ndarray
    width       : np.ndarray
    length      : np.ndarray = field(init=False)
    h           : Optional[np.ndarray] = None
    k           : Optional[np.ndarray] = None
    l           : Optional[np.ndarray] = None
    hkl_index   : Optional[np.ndarray] = None

    def __post_init__(self):
        self.length = np.sqrt((self.x1 - self.x0)**2 + (self.y1 - self.y0)**2)

    def __getitem__(self, attr: str) -> Optional[np.ndarray]:
        return self.__getattribute__(attr)

    def keys(self) -> List[str]:
        return [attr for attr in self.__dataclass_fields__.keys()
                if self.__getitem__(attr) is not None]

    def values(self) -> ValuesView:
        return dict(self).values()

    def items(self) -> ItemsView:
        return dict(self).items()

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(dict(self))

    def to_numpy(self) -> np.ndarray:
        return np.stack(tuple(self.values()), axis=1)

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
            if attr in self.keys():
                pattern[attr] = self.__getattribute__(attr)[idx]
        return pattern

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
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @dict_to_object
    def update_lsd(self, scale: float=0.9, sigma_scale: float=0.9,
                   log_eps: float=0., ang_th: float=60.0, density_th: float=0.5,
                   quant: float=2e-2) -> Dict:
        return {'lsd_obj': LSD(scale=scale, sigma_scale=sigma_scale, log_eps=log_eps,
                               ang_th=ang_th, density_th=density_th, quant=quant)}

    def to_dataframe(self, concatenate: bool=True) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        if self.streaks is None:
            raise ValueError("No 'streaks' specified inside the container.")

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
        if self.streaks is None:
            raise ValueError("No 'streaks' specified inside the container.")

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

    @dict_to_object
    def generate_streak_data(self, vmin: float, vmax: float,
                             size: Union[Tuple[int, ...], int]=(1, 3, 3)) -> StreakDetector:
        streak_data = median_filter(self.data, size=size, num_threads=self.num_threads)
        streak_data = np.divide(np.clip(streak_data, vmin, vmax) - vmin, vmax - vmin)
        return {'streak_data': streak_data}

    @dict_to_object
    def detect(self, cutoff: float, filter_threshold: float,
               group_threshold: float=0.7, dilation: float=0.0, n_group: int=2) -> StreakDetector:
        out_dict = self.lsd_obj.detect(self.streak_data, cutoff=cutoff,
                                       filter_threshold=filter_threshold,
                                       group_threshold=group_threshold,
                                       n_group=n_group, dilation=dilation,
                                       num_threads=self.num_threads)
        return {'streaks': {self.frames[idx]: Streaks(*np.around(lines[:, :5], 2).T)
                            for idx, lines in out_dict['lines'].items()}}

    @dict_to_object
    def generate_bgd_mask(self, bgd_dilation: float=8.0) -> StreakDetector:
        bgd_mask = self.draw(dilation=bgd_dilation)
        return {'bgd_dilation': bgd_dilation, 'bgd_mask': bgd_mask}

    @dict_to_object
    def update_streak_data(self) -> StreakDetector:
        if self.streaks is None:
            raise ValueError("No 'streaks' specified inside the container.")

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
        return {'streak_data': streak_data}

    def draw(self, max_val: int=1, dilation: float=0.0) -> np.ndarray:
        if self.streaks is None:
            raise ValueError("No 'streaks' specified inside the container.")

        streaks = {key: val.to_numpy() for key, val in self.streaks.items()}
        mask = np.zeros(self.data.shape, dtype=np.uint32)
        return draw_lines_stack(mask, lines=streaks, max_val=max_val, dilation=dilation,
                                num_threads=self.num_threads)
