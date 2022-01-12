"""CXI protocol (:class:`cbclib.CXIProtocol`) is a helper class for a :class:`cbclib.STData`
data container, which tells it where to look for the necessary data fields in a CXI
file. The class is fully customizable so you can tailor it to your particular data
structure of CXI file.

Examples:
    Generate the default built-in CXI protocol as follows:

    >>> import cbclib as cbc
    >>> cbc.CXIProtocol.import_default()
    {'config': {'float_precision': 'float64'}, 'datatypes': {'basis_vectors': 'float',
    'data': 'uint', 'defocus_x': 'float', '...': '...'}, 'default_paths': {'basis_vectors':
    '/speckle_tracking/basis_vectors', 'data': '/entry/data/data', 'defocus_x':
    '/speckle_tracking/defocus_x', '...': '...'}, 'is_data': {'basis_vectors': 'False',
    'data': 'True', 'defocus_x': 'False', '...': '...'}}
"""
from __future__ import annotations
from multiprocessing import Pool
import os
from configparser import ConfigParser
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union
import h5py
import numpy as np
from tqdm.auto import tqdm
from .ini_parser import ROOT_PATH, INIParser

CXI_PROTOCOL = os.path.join(ROOT_PATH, 'config/cxi_protocol.ini')

class CXIProtocol(INIParser):
    """CXI protocol class. Contains a CXI file tree path with
    the paths written to all the data attributes necessary for
    the Speckle Tracking algorithm, corresponding attributes'
    data types, and floating point precision.

    Attributes:
        config : Protocol configuration.
        datatypes : Dictionary with attributes' datatypes. 'float', 'int', 'uint',
            or 'bool' are allowed.
        is_data : Dictionary with the flags if the attribute is of data type.
            Data type is 2- or 3-dimensional and has the same data shape
            as `data`.
        load_paths : Dictionary with attributes' CXI default file paths.
    """
    known_types = {'int': np.integer, 'bool': np.bool, 'float': np.floating, 'str': str,
                   'uint': np.unsignedinteger}
    known_ndims = {'stack': 3, 'frame': 2, 'sequence': 1, 'scalar': 0}
    attr_dict = {'datatypes': ('ALL', ), 'load_paths': ('ALL', ), 'kinds': ('ALL', )}
    fmt_dict = {'datatypes': 'str', 'load_paths': 'str', 'kinds': 'str'}

    def __init__(self, datatypes: Dict[str, str], load_paths: Dict[str, Union[str, List[str]]],
                 kinds: Dict[str, str]) -> None:
        """
        Args:
            datatypes : Dictionary with attributes' datatypes. 'float', 'int', 'uint',
                or 'bool' are allowed.
            load_paths : Dictionary with attributes path in the CXI file.
            kinds : Dictionary with the flags if the attribute is of data type.
                Data type is 2- or 3-dimensional and has the same data shape
                as `data`.
            float_precision : Choose between 32-bit floating point precision ('float32')
                or 64-bit floating point precision ('float64').
        """
        load_paths = {attr: self.str_to_list(val)
                      for attr, val in load_paths.items() if attr in datatypes}
        kinds = {attr: val for attr, val in kinds.items() if attr in datatypes}
        super(CXIProtocol, self).__init__(datatypes=datatypes, load_paths=load_paths,
                                          kinds=kinds)

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
    def import_default(cls, datatypes: Optional[Dict[str, str]]=None,
                       load_paths: Optional[Dict[str, Union[str, List[str]]]]=None,
                       kinds: Optional[Dict[str, str]]=None) -> CXIProtocol:
        """Return the default :class:`CXIProtocol` object. Extra arguments
        override the default values if provided.

        Args:
            datatypes : Dictionary with attributes' datatypes. 'float', 'int', 'uint',
                or 'bool' are allowed.
            load_paths : Dictionary with attributes path in the CXI file.
            kinds : Dictionary with the flags if the attribute is of data type.
                Data type is 2- or 3-dimensional and has the same data shape
                as `data`.

        Returns:
            A :class:`CXIProtocol` object with the default parameters.
        """
        return cls.import_ini(CXI_PROTOCOL, datatypes, load_paths, kinds)

    @classmethod
    def import_ini(cls, ini_file: str, datatypes: Optional[Dict[str, str]]=None,
                   load_paths: Optional[Dict[str, Union[str, List[str]]]]=None,
                   kinds: Optional[Dict[str, str]]=None) -> CXIProtocol:
        """Initialize a :class:`CXIProtocol` object class with an
        ini file.

        Args:
            ini_file : Path to the ini file. Load the default CXI protocol if None.
            datatypes : Dictionary with attributes' datatypes. 'float', 'int', 'uint',
                or 'bool' are allowed. Initialized with `ini_file` if None.
            load_paths : Dictionary with attributes path in the CXI file. Initialized
                with `ini_file` if None.
            kinds : Dictionary with the flags if the attribute is of data type.
                Data type is 2- or 3-dimensional and has the same data shape
                as `data`. Initialized with `ini_file` if None.

        Returns:
            A :class:`CXIProtocol` object with all the attributes imported
            from the ini file.
        """
        kwargs = cls._import_ini(ini_file)
        if not datatypes is None:
            kwargs['datatypes'].update(**datatypes)
        if not load_paths is None:
            kwargs['load_paths'].update(**load_paths)
        if not kinds is None:
            kwargs['kinds'].update(**kinds)
        return cls(datatypes=kwargs['datatypes'], load_paths=kwargs['load_paths'],
                   kinds=kwargs['kinds'])

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

    def parser_from_template(self, path: str) -> ConfigParser:
        """Return a :class:`configparser.ConfigParser` object using
        an ini file template.

        Args:
            path : Path to the ini file template.

        Returns:
            Parser object with the attributes populated according
            to the protocol.
        """
        ini_template = ConfigParser()
        ini_template.read(path)
        parser = ConfigParser()
        for section in ini_template:
            parser[section] = {option: ini_template[section][option].format(**self.default_paths)
                               for option in ini_template[section]}
        return parser

    def __iter__(self) -> Iterator:
        return self.datatypes.__iter__()

    def __contains__(self, attr: str) -> bool:
        return attr in self.datatypes

    def get_load_paths(self, attr: str, value: Optional[List[str]]=None) -> List[str]:
        """Return the atrribute's default path in the CXI file.
        Return `value` if `attr` is not found.

        Args:
            attr : The attribute to look for.
            value : Value which is returned if the `attr` is not found.

        Returns:
            Attribute's cxi file path.
        """
        return self.load_paths.get(attr, value)

    def get_dtype(self, attr: str, dtype: Optional[str]='float') -> type:
        """Return the attribute's data type. Return `dtype` if the attribute's
        data type is not found.

        Args:
            attr : The data attribute.
            dtype : Data type which is returned if the attribute's data type
                is not found.

        Returns:
            Attribute's data type.
        """
        return self.known_types.get(self.datatypes.get(attr, dtype))

    def get_kind(self, attr: str, value: str='scalar') -> str:
        """Return if the attribute is of data type. Data type is 2- or
        3-dimensional and has the same data shape as `data`.

        Args:
            attr : The data attribute.
            value : value which is returned if the `attr` is not found.

        Returns:
            True if `attr` is of data type.
        """
        return self.kinds.get(attr, value)        

    def get_ndim(self, attr: str, value: int=0) -> int:
        return self.known_ndims.get((self.get_kind(attr)), value)

    def cast(self, attr: str, array: np.ndarray):
        dtype = self.get_dtype(attr)
        if np.issubdtype(array.dtype, dtype):
            return array
        return np.asarray(array, dtype=dtype)

    @staticmethod
    def read_dataset_shapes(cxi_path: str, cxi_file: h5py.File) -> Dict[str, Tuple]:
        """Visit recursively all the underlying datasets and return
        their names and shapes.

        Args:
            cxi_path : Path of the HDF5 group.
            cxi_obj: Group object.

        Returns:
            List of all the datasets and their shapes inside `cxi_obj`.
        """
        shapes = {}

        def caller(sub_path, obj):
            if isinstance(obj, h5py.Dataset):
                shapes[os.path.join(cxi_path, sub_path)] = obj.shape

        if cxi_path in cxi_file:
            cxi_obj = cxi_file[cxi_path]
            if isinstance(cxi_obj, h5py.Dataset):
                shapes[cxi_path] = cxi_obj.shape
            elif isinstance(cxi_obj, h5py.Group):
                cxi_obj.visititems(caller)
            else:
                raise ValueError(f"Invalid CXI object at '{cxi_path:s}'")

        cxi_obj.visititems(caller)
        return shapes

    def read_attribute_shapes(self, attr: str, data_file: str):
        with h5py.File(data_file, 'r') as cxi_file:
            cxi_path = self.find_path(attr, cxi_file)
            return self.read_dataset_shapes(cxi_path, cxi_file)

    def read_attribute_indices(self, attr, data_files):
        files, cxi_paths, fidxs = [], [], []
        kind = self.get_kind(attr)

        for data_file in data_files:
            shapes = self.read_attribute_shapes(attr, data_file)
            for cxi_path, shape in shapes.items():
                if len(shape) != self.get_ndim(attr):
                    err_txt = f'Dataset at {data_file}: {cxi_path} has invalid shape: {str(shape)}'
                    raise ValueError(err_txt)

                if kind in ['stack', 'sequence']:
                    files.extend(np.repeat(data_file, shape[0]).tolist())
                    cxi_paths.extend(np.repeat(cxi_path, shape[0]).tolist())
                    fidxs.extend(np.arange(shape[0]).tolist())
                if kind  == 'frame':
                    files.append(data_file)
                    cxi_paths.append(cxi_path)
                    fidxs.append(tuple())

        return np.array([files, cxi_paths, fidxs], dtype=object).T

class CXIStore():
    def __init__(self, data_files: Union[str, List[str]],
                 protocol: CXIProtocol=CXIProtocol.import_default()) -> None:
        data_files = protocol.str_to_list(data_files)

        for data_file in data_files:
            if not os.path.isfile(data_file):
                raise ValueError(f'There is no file under the given path: {data_file}')

        self.data_files, self.protocol = data_files, protocol
        self._indices = self.read_indices()

        if 'data' not in self._indices:
            raise ValueError("Data files don't contain 'data' attribute")

    def read_indices(self):
        indices = {}
        for attr in self.protocol:
            kind = self.protocol.get_kind(attr)
            if kind in ['stack', 'sequence']:
                indices[attr] = self.read_attribute_indices(attr)
            if kind == 'frames':
                indices[attr] = self.read_attribute_indices(attr)[0]
        return indices

    def indices(self):
        return np.arange(self._indices['data'].shape[0])

    def keys(self):
        return self._indices.keys()

    def values(self):
        return self._indices.values()

    def items(self):
        return self._indices.items()

    @staticmethod
    def _read_chunk(index: np.ndarray) -> np.ndarray:
        with h5py.File(index[0], 'r') as cxi_file:
            return cxi_file[index[1]][index[2]]

    def _load_stack(self, attr: str, indices: Iterable[int], processes: int=1, verbose: bool=True) -> np.ndarray:
        stack = []

        with Pool(processes=processes) as pool:
            for frame in tqdm(pool.imap(type(self)._read_chunk,
                                        self._indices[attr][indices]),
                              total=self.indices()[indices].size, disable=not verbose):
                stack.append(frame)

        return self.protocol.cast(attr, np.stack(stack, axis=0))

    def _load_frame(self, attr: str) -> np.ndarray:
        return self.protocol.cast(attr, self._read_chunk(self._indices[attr]))

    def _load_sequence(self, attr: str, indices: Iterable[int]) -> np.ndarray:
        sequence = []

        for index in self._indices[attr][indices]):
            with h5py.File(index[0], 'r') as cxi_file:
                sequence.append(cxi_file[index[1]][index[2]])

        return self.protocol.cast(attr, np.array(sequence))

    def load_attribute(self, attr: str, indices: Optional[Iterable[int]]=None, processes: int=1,
                       verbose: bool=True) -> np.ndarray:
        kind = self.protocol.get_kind(attr)

        if kind == 'stack':
            return self._load_stack()
