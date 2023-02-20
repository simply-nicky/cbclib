<<<<<<< HEAD
from __future__ import annotations
from multiprocessing import cpu_count
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
from weakref import ref, ReferenceType
import numpy as np
import pandas as pd
from .cxi_protocol import CXIStore
from .data_container import DataContainer, dict_to_object
from .bin import (subtract_background, project_effs, median, median_filter, LSD,
                  maximum_filter, normalize_streak_data, draw_line_indices_aa)
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
    attr_set = {'input_files'}
    init_set = {'background', 'cor_data', 'data', 'good_frames', 'frames', 'mask',
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
    streaks:        Optional[np.ndarray]
    tilts:          Optional[np.ndarray]
    translations:   Optional[np.ndarray]
    whitefield:     Optional[np.ndarray]

    def __init__(self, input_files: CXIStore, output_file: Optional[CXIStore]=None,
                 transform: Optional[Transform]=None, **kwargs):
        super(CrystData, self).__init__(input_files=input_files, output_file=output_file,
                                        transform=transform, **kwargs)

        self._init_functions(num_threads=lambda: np.clip(1, 64, cpu_count()))
        if self._isdata:
            self._init_functions(good_frames=lambda: np.where(self.data.sum(axis=(1, 2)) > 0)[0],
                                 mask=self._mask)
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
=======
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
from .bin import (subtract_background, project_effs, median, median_filter, LSD,
                  normalise_pattern, refine_pattern, draw_line_mask)

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
    input_file  : Optional[CXIStore] = None
    transform   : Optional[Transform] = None
    num_threads : int = cpu_count()
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
    def protocol(self) -> Optional[CXIProtocol]:
        if self.input_file is not None:
            return self.input_file.protocol
        if self.output_file is not None:
            return self.output_file.protocol
        return None
>>>>>>> dev-dataclass

    @property
    def shape(self) -> Tuple[int, int, int]:
        shape = [0, 0, 0]
        for attr, data in self.items():
<<<<<<< HEAD
            if attr in self.input_files.protocol and data is not None:
                kind = self.input_files.protocol.get_kind(attr)
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
=======
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
>>>>>>> dev-dataclass

        Args:
            x : A set of transformed x coordinates.
            y : A set of transformed y coordinates.

<<<<<<< HEAD
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
        for attr in attributes:
            data = self.get(attr)
            if attr in self.input_files.protocol and data is not None:
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
=======
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
>>>>>>> dev-dataclass

        if attributes is None:
            attributes = list(self.contents())

<<<<<<< HEAD
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

        self.streak_data = np.zeros(self.shape, dtype=detector.dtypes['streak_data'])
        self.streak_data[self.good_frames] = detector.streak_data

    @dict_to_object
    def import_whitefield(self, whitefield: np.ndarray) -> CrystData:
=======
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
>>>>>>> dev-dataclass
        """Return a new :class:`CrystData` object with the new
        whitefield.

        Args:
            whitefield : New whitefield array.

        Raises:
            ValueError : If the whitefield shape is incompatible with the data.
<<<<<<< HEAD

        Returns:
            New :class:`CrystData` object with the updated `whitefield`.
        """
        if whitefield.shape != self.shape[1:]:
            raise ValueError('whitefield and data have incompatible shapes: '\
                             f'{whitefield.shape:s} != {self.shape[1:]:s}')
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
=======
            ValueError : If there is no ``data`` inside the container.

        Returns:
            New :class:`CrystData` object with the updated ``whitefield``.
        """
        raise self._no_data_exc

    def update_mask(self: C, method: str='range-bad', pmin: float=0.0, pmax: float=99.99,
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
            ValueError : If ``method`` keyword is invalid.
            ValueError : If ``update`` keyword is invalid.
            ValueError : If ``vmin`` is larger than ``vmax``.
            ValueError : If ``pmin`` is larger than ``pmax``.

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
>>>>>>> dev-dataclass
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
    input_file  : Optional[CXIStore] = None
    transform   : Optional[Transform] = None
    num_threads : int = cpu_count()
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
        x0, y0 = self.forward_points(x=setup.pupil_roi[2], y=setup.pupil_roi[0])
        x1, y1 = self.forward_points(x=setup.pupil_roi[3], y=setup.pupil_roi[1])
        return self.mask_region((int(y0 - padding), int(y1 + padding),
                                 int(x0 - padding), int(x1 + padding)))

    def import_mask(self: C, mask: np.ndarray, update: str='reset') -> C:
        if mask.shape != self.shape[1:]:
            raise ValueError('mask and data have incompatible shapes: '\
                             f'{mask.shape:s} != {self.shape[1:]:s}')

        if update == 'reset':
<<<<<<< HEAD
            data = self.data[self.good_frames]
        elif update == 'multiply':
            data = self.data[self.good_frames] * self.mask[self.good_frames]
=======
            return self.replace(mask=mask)
        if update == 'multiply':
            return self.replace(mask=mask * self.mask)

        raise ValueError(f'Invalid update keyword: {update:s}')

    def import_whitefield(self, whitefield: np.ndarray) -> CrystDataFull:
        if sum(self.shape[1:]) and whitefield.shape != self.shape[1:]:
            raise ValueError('whitefield and data have incompatible shapes: '\
                             f'{whitefield.shape} != {self.shape[1:]}')
        return CrystDataFull(**dict(self, whitefield=whitefield))

    def update_mask(self, method: str='range-bad', pmin: float=0.0, pmax: float=99.99,
                    vmin: int=0, vmax: int=65535, update: str='reset') -> C:
        if vmin >= vmax:
            raise ValueError('vmin must be less than vmax')
        if pmin >= pmax:
            raise ValueError('pmin must be less than pmax')

        if update == 'reset':
            data = self.data
        elif update == 'multiply':
            data = self.data * self.mask
>>>>>>> dev-dataclass
        else:
            raise ValueError(f'Invalid update keyword: {update:s}')

        mask = np.zeros(self.shape, dtype=bool)
        if method == 'no-bad':
<<<<<<< HEAD
            mask[self.good_frames] = True
=======
            mask = np.ones(self.shape, dtype=bool)
>>>>>>> dev-dataclass
        elif method == 'range-bad':
            mask[self.good_frames] = (data >= vmin) & (data < vmax)
        elif method == 'perc-bad':
            average = median_filter(data, (1, 3, 3), num_threads=self.num_threads)
            offsets = (data.astype(np.int32) - average.astype(np.int32))
<<<<<<< HEAD
            mask[self.good_frames] = (offsets >= np.percentile(offsets, pmin)) & \
                                     (offsets <= np.percentile(offsets, pmax))
=======
            mask = (offsets >= np.percentile(offsets, pmin)) & \
                   (offsets <= np.percentile(offsets, pmax))
>>>>>>> dev-dataclass
        else:
            ValueError('invalid method argument')

        if update == 'reset':
<<<<<<< HEAD
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

        dtypes = StreakDetector.dtypes
        data = np.asarray(self.cor_data[self.good_frames], order='C', dtype=dtypes['data'])
        frames = dict(zip(self.frames[self.good_frames], self.good_frames))
        if not data.size:
            raise ValueError('No good frames in the stack')

        if import_contents and self.streak_data is not None:
            streak_data = np.asarray(self.streak_data[self.good_frames], order='C',
                                     dtype=dtypes['streak_data'])
            return StreakDetector(parent=ref(self), data=data, frames=frames,
                                  num_threads=self.num_threads, streak_data=streak_data)

        return StreakDetector(parent=ref(self), data=data, frames=frames,
                              num_threads=self.num_threads)

class StreakDetector(DataContainer):
    attr_set: Set[str] = {'parent', 'data', 'frames', 'num_threads'}
    init_set: Set[str] = {'bgd_dilation', 'bgd_mask', 'lsd_obj', 'streak_data', 'streak_dilation',
                          'streak_mask', 'streaks'}
    dtypes: Dict[str, np.dtype] = {'bgd_mask': np.uint32, 'data': np.float32, 'frames': np.uint32,
                                   'streak_data': np.float32, 'streak_mask': np.uint32,
                                   'streaks': np.float32}
    footprint: np.ndarray = np.array([[[False, False,  True, False, False],
                                       [False,  True,  True,  True, False],
                                       [ True,  True,  True,  True,  True],
                                       [False,  True,  True,  True, False],
                                       [False, False,  True, False, False]]])
    txt_header: str = 'Frame  x0, pix  y0, pix  x1, pix  y1, pix  w, pix'
    txt_fmt = ('%7d', '%8.2f', '%8.2f', '%8.2f', '%8.2f', '%7.2f')

    parent: ReferenceType[CrystData]
    lsd_obj: LSD

    bgd_dilation : int
    bgd_mask : np.ndarray
    data: np.ndarray
    frames: Dict[int, int]
    num_threads: int
    streak_dilation : int
    streak_data: np.ndarray
    streak_mask: np.ndarray
    streaks: Dict[int, np.ndarray]
    txt_fmt: Tuple[str, str, str, str, str, str]

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
                                                 ang_th=60.0, density_th=0.5, quant=2e-2))

        self._init_attributes()

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.data.shape

    @staticmethod
    def frame_dict() -> Dict[str, List]:
        return {'frames': [], 'streaks': [], 'x': [], 'y': [], 'I': [], 'bgd': []}

    @dict_to_object
    def update_lsd(self, scale: float=0.9, sigma_scale: float=0.9,
                   log_eps: float=0., ang_th: float=60.0, density_th: float=0.5,
                   quant: float=2e-2) -> StreakDetector:
        return {'lsd_obj': LSD(scale=scale, sigma_scale=sigma_scale, log_eps=log_eps,
                               ang_th=ang_th, density_th=density_th, quant=quant)}

    @dict_to_object
    def load_txt(self, path: str) -> StreakDetector:
        txt_arr = np.loadtxt(path)
        frames, idxs = np.unique(txt_arr[:, 0], return_inverse=True)
        streaks = {}
        for idx in range(frames.size):
            streaks[int(frames[idx])] = txt_arr[idxs == idx][:, 1:].astype(self.dtypes['streaks'])
        return {'streaks': streaks}

    def save_txt(self, path: str) -> None:
        if self.streaks is None:
            raise ValueError("No 'streaks' specified inside the container.")
        txt_list = []
        for frame, streaks in list(self.streaks.items()):
            txt_arr = np.concatenate([frame * np.ones((streaks.shape[0], 1),
                                                      dtype=streaks.dtype),
                                     streaks[:, :5]], axis=1)
            txt_list.append(txt_arr)
        np.savetxt(path, np.concatenate(txt_list, axis=0), fmt=self.txt_fmt,
                   header=self.txt_header)

    def to_dataframe(self) -> pd.DataFrame:
        if self.streaks:
            frame_dict = self.frame_dict()
            for frame, index in self.frames.items():
                idxs = draw_line_indices_aa(lines=self.streaks[frame], shape=self.shape[1:],
                                            max_val=1)
                frame_dict['frames'].append(frame * np.ones(idxs.shape[0],
                                                            dtype=self.dtypes['frames']))
                frame_dict['streaks'].append(idxs[:, 0])
                frame_dict['x'].append(idxs[:, 1])
                frame_dict['y'].append(idxs[:, 2])
                frame_dict['I'].append(self.parent().cor_data[index][idxs[:, 2], idxs[:, 1]])
                frame_dict['bgd'].append(self.parent().background[index][idxs[:, 2], idxs[:, 1]])

            return pd.DataFrame({key: np.concatenate(val) for key, val in frame_dict.items()})

        raise ValueError("No 'streaks' specified inside the container.")

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
        return {'streaks': {self.frames[idx]: np.around(lines[:, :5], 2)
                            for idx, lines in out_dict['lines'].items()}}

    @dict_to_object
    def update_streak_mask(self, streak_dilation: int=6, bgd_dilation: int=14) -> StreakDetector:
        if self.streaks:
            streak_mask = np.zeros(self.data.shape, dtype=self.dtypes['streak_mask'])
            streak_mask = self.lsd_obj.draw_lines(streak_mask, self.streaks,
                                                dilation=streak_dilation,
                                                num_threads=self.num_threads)
            bgd_mask = np.zeros(self.data.shape, dtype=self.dtypes['bgd_mask'])
            bgd_mask = self.lsd_obj.draw_lines(bgd_mask, self.streaks,
                                            dilation=bgd_dilation,
                                            num_threads=self.num_threads)
            return {'bgd_dilation': bgd_dilation, 'bgd_mask': bgd_mask,
                    'streak_dilation': streak_dilation, 'streak_mask': streak_mask}

        raise AttributeError("No 'streaks' specified inside the container.")

    @dict_to_object
    def update_streak_data(self) -> StreakDetector:
        if self.streak_mask is None or self.bgd_mask is None:
            raise AttributeError("'streak_mask' and 'bgd_mask' must be generated before.")

        divisor = self.data
        for _ in range(self.streak_dilation // 2):
            divisor = maximum_filter(divisor, mask=self.streak_mask, footprint=self.footprint,
                                     num_threads=self.num_threads)

        bgd = self.data * (self.bgd_mask - self.streak_mask)
        for _ in range(self.bgd_dilation // 2):
            bgd = median_filter(bgd, mask=self.bgd_mask, good_data=bgd,
                                footprint=self.footprint, num_threads=self.num_threads)

        streak_data = normalize_streak_data(data=self.data, bgd=bgd, divisor=divisor,
                                            num_threads=self.num_threads)
        return {'streak_data': streak_data}
=======
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
    input_file  : Optional[CXIStore] = None
    transform   : Optional[Transform] = None
    num_threads : int = cpu_count()
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
        x0, y0 = self.forward_points(x=setup.pupil_roi[2], y=setup.pupil_roi[0])
        x1, y1 = self.forward_points(x=setup.pupil_roi[3], y=setup.pupil_roi[1])

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
    profile         : ClassVar[str] = 'gauss'
    _no_streaks_exc : ClassVar[ValueError] = ValueError('No streaks in the container')

    data            : np.ndarray
    frames          : np.ndarray
    num_threads     : int
    parent          : ReferenceType[CrystData]
    streaks         : Dict[int, Streaks]

    patterns        : Optional[np.ndarray] = None

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
                * `gauss` : Gaussian function profile.

        Raises:
            ValueError : If there is no ``streaks`` inside the container.

        Returns:
            A pattern mask.
        """
        raise self._no_streaks_exc

    def export_streaks(self):
        """Export ``streak_mask`` to the parent :class:`cbclib.CrystData` data container.

        Raises:
            ValueError : If there is no ``streaks`` inside the container.
            ValueError : If there is no ``streak_mask`` inside the container.
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
            * `p` : Normalised pattern values.
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

    def update_patterns(self: D, dilations: Tuple[float, float, float]=(1.0, 3.0, 7.0)) -> D:
        """Return a new detector object with updated normalised CBC patterns. The image is
        segmented into two region around each reflection to calculate the local background
        and local peak intensity. The estimated values are used to normalise each diffraction
        streak separately.

        Args:
            dilations : A tuple of three dilations (`d0`, `d1`, `d2`) in pixels of the streak
                mask that is used to define the inner and outer streak zones:

                * The inner zone is based on the mask dilated by `d0`.
                * The outer zone is based on the difference between a mask dilated by `d2` and
                  by `d1`.

        Returns:
            A new detector object with updated ``patterns``.
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
    def draw(self, max_val: int=1, dilation: float=0.0, profile: str='tophat') -> np.ndarray:
        streaks = [streaks.to_numpy() for streaks in self.streaks.values()]
        return draw_line_mask(self.shape, lines=streaks, max_val=max_val, dilation=dilation,
                              profile=profile, num_threads=self.num_threads)

    def export_streaks(self):
        if self.parent() is None:
            raise ValueError('Invalid parent: the parent data container was deleted')

        streak_mask = np.zeros(self.parent().shape, dtype=bool)
        streak_mask[self.parent().good_frames] = self.draw()
        self.parent().streak_mask = streak_mask

    def export_table(self, dilation: float=0.0, concatenate: bool=True) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        if self.parent() is None:
            raise ValueError('Invalid parent: the parent data container was deleted')
        if self.patterns is None:
            raise ValueError('No patterns in the container')

        dataframes, frames = [], []
        for idx, streaks in self.streaks.items():
            df = streaks.pattern_dataframe(shape=self.shape[1:], dilation=dilation, profile=self.profile)
            df['frames'] = self.frames[idx] * np.ones(df.shape[0], dtype=df['index'].dtype)

            index = self.parent().good_frames[idx]
            df = df[self.parent().mask[index, df['y'], df['x']]]
            df['p'] = self.patterns[idx, df['y'], df['x']]
            df['I_raw'] = self.parent().data[index, df['y'], df['x']]
            df['bgd'] = self.parent().background[index, df['y'], df['x']]
            df['x'], df['y'] = self.parent().backward_points(df['x'], df['y'])

            frames.append(self.frames[idx])
            dataframes.append(df)

        dataframes = [df for _, df in sorted(zip(frames, dataframes))]

        if concatenate:
            return pd.concat(dataframes, ignore_index=True)
        return dataframes

    def refine_streaks(self: D, dilation: float=0.0) -> D:
        streaks = {key: val.to_numpy() for key, val in self.streaks.items()}
        streaks = refine_pattern(inp=self.data, lines=streaks, dilation=dilation,
                                 num_threads=self.num_threads)
        return self.replace(streaks={idx: Streaks(*(lines.T)) for idx, lines in streaks.items()})

    def update_patterns(self: D, dilations: Tuple[float, float, float]=(1.0, 3.0, 7.0)) -> D:
        streaks = {key: val.to_numpy() for key, val in self.streaks.items()}
        patterns = normalise_pattern(inp=self.data, lines=streaks, dilations=dilations,
                                     profile=self.profile, num_threads=self.num_threads)
        return self.replace(patterns=patterns)

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
        patterns : Normalized diffraction patterns.
    """
    data            : np.ndarray
    frames          : np.ndarray
    num_threads     : int
    parent          : ReferenceType[CrystData]
    lsd_obj         : LSD = field(default=LSD(0.9, 0.9, 0.0, 45.0, 0.5, 2e-2))
    streaks         : Dict[int, Streaks] = field(default_factory=dict)

    patterns        : Optional[np.ndarray] = None

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
        if self.patterns is None:
            raise ValueError('No pattern in the container')

        out_dict = self.lsd_obj.detect(self.patterns, cutoff=cutoff,
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

    def generate_patterns(self: D, vmin: float, vmax: float,
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
        patterns = median_filter(self.data, size=size, num_threads=self.num_threads)
        patterns = np.divide(np.clip(patterns, vmin, vmax) - vmin, vmax - vmin)
        return self.replace(patterns=np.asarray(patterns, dtype=np.float32))

@dataclass
class LSDetectorFull(DetectorFull, LSDetector):
    data            : np.ndarray
    frames          : np.ndarray
    num_threads     : int
    parent          : ReferenceType[CrystData]
    lsd_obj         : LSD = field(default=LSD(0.9, 0.9, 0.0, 60.0, 0.5, 2e-2))
    streaks         : Dict[int, Streaks] = field(default_factory=dict)

    patterns        : Optional[np.ndarray] = None

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
        patterns : Normalized diffraction patterns.=
    """
    data            : np.ndarray
    frames          : np.ndarray
    num_threads     : int
    parent          : ReferenceType[CrystData]
    models          : Dict[int, CBDModel]
    streaks         : Dict[int, Streaks] = field(default_factory=dict)

    patterns        : Optional[np.ndarray] = None

    def detect(self, hkl: np.ndarray, width: float=4.0, threshold: float=1.0) -> ModelDetectorFull:
        """Perform the streak detection based on prediction. Generate a predicted pattern and
        filter out all the streaks, which signal-to-noise ratio is below the ``threshold``.

        Args:
            hkl : A set of reciprocal lattice points used in the detection.
            width : Difrraction streak width in pixels of a predicted pattern.
            threshold : SNR threshold.

        Returns:
            New :class:`cbclib.ModelDetector` streak detector with updated ``streaks``.
        """
        if self.parent() is None:
            raise ValueError('Invalid parent: the parent data container was deleted')

        streaks = {}
        for idx, model in self.models.items():
            index = self.parent().good_frames[idx]
            stks = model.filter_streaks(hkl=hkl[model.filter_hkl(hkl)], threshold=threshold,
                                        signal=self.parent().cor_data[index], profile=self.profile,
                                        background=self.parent().background[index],  width=width,
                                        num_threads=self.num_threads)
            streaks[idx] = stks
        return ModelDetectorFull(**dict(self, streaks=streaks))

@dataclass
class ModelDetectorFull(DetectorFull, ModelDetector):
    data            : np.ndarray
    frames          : Dict[int, int]
    num_threads     : int
    parent          : ReferenceType[CrystData]

    models          : Dict[int, CBDModel]
    indices         : Dict[int, int] = field(default_factory=dict)
    streaks         : Dict[int, Streaks] = field(default_factory=dict)

    patterns        : Optional[np.ndarray] = None
>>>>>>> dev-dataclass
