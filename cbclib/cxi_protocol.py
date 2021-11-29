"""Examples
--------
Generate the default built-in CXI protocol as follows:

...
"""
import os
import configparser
import h5py
import numpy as np
from tqdm.auto import tqdm
from .ini_parser import ROOT_PATH, INIParser
from .data_processing import CrystData

CXI_PROTOCOL = os.path.join(ROOT_PATH, 'config/cxi_protocol.ini')

class CXIProtocol(INIParser):
    """CXI protocol class. Contains a CXI file tree path with
    the paths written to all the data attributes necessary for
    the Speckle Tracking algorithm, corresponding attributes'
    data types, and floating point precision.

    Parameters
    ----------
    datatypes : dict, optional
        Dictionary with attributes' datatypes. 'float', 'int', or 'bool'
        are allowed.
    default_paths : dict, optional
        Dictionary with attributes path in the CXI file.
    is_data : dict, optional
        Dictionary with the flags if the attribute is of data type.
        Data type is 2- or 3-dimensional and has the same data shape
        as `data`.

    Attributes
    ----------
    datatypes : dict
        Dictionary with attributes' datatypes. 'float', 'int', or 'bool'
        are allowed.
    default_paths : dict
        Dictionary with attributes' CXI default file paths.
    is_data : dict
        Dictionary with the flags if the attribute is of data type.
        Data type is 2- or 3-dimensional and has the same data shape
        as `data`.

    See Also
    --------
    cxi_protocol : Full list of data attributes and configuration
        parameters.
    """
    known_types = {'int': np.int32, 'bool': np.bool, 'float': np.float32, 'str': str,
                   'uint': np.uint32}
    attr_dict = {'datatypes': ('ALL', ), 'is_data': ('ALL', ),
                 'default_paths': ('ALL', )}
    fmt_dict = {'datatypes': 'str', 'default_paths': 'str', 'is_data': 'str'}

    def __init__(self, datatypes=None, default_paths=None, is_data=None):
        if datatypes is None:
            datatypes = self._import_ini(CXI_PROTOCOL)['datatypes']
        if default_paths is None:
            default_paths = self._import_ini(CXI_PROTOCOL)['default_paths']
        if is_data is None:
            is_data = self._import_ini(CXI_PROTOCOL)['is_data']

        datatypes = {attr: val for attr, val in datatypes.items() if attr in default_paths}
        is_data = {attr: val for attr, val in is_data.items() if attr in datatypes}
        super(CXIProtocol, self).__init__(datatypes=datatypes, default_paths=default_paths,
                                          is_data=is_data)

    @classmethod
    def import_default(cls, datatypes=None, default_paths=None, is_data=None):
        """Return the default :class:`CXIProtocol` object. Extra arguments
        override the default values if provided.

        Parameters
        ----------
        datatypes : dict, optional
            Dictionary with attributes' datatypes. 'float', 'int', or 'bool'
            are allowed.
        default_paths : dict, optional
            Dictionary with attributes path in the CXI file.
        is_data : dict, optional
            Dictionary with the flags if the attribute is of data type.
            Data type is 2- or 3-dimensional and has the same data shape
            as `data`.

        Returns
        -------
        CXIProtocol
            A :class:`CXIProtocol` object with the default parameters.

        See Also
        --------
        cxi_protocol : more details about the default CXI protocol.
        """
        return cls.import_ini(CXI_PROTOCOL, datatypes, default_paths, is_data)

    @classmethod
    def import_ini(cls, ini_file, datatypes=None, default_paths=None, is_data=None):
        """Initialize a :class:`CXIProtocol` object class with an
        ini file.

        Parameters
        ----------
        ini_file : str
            Path to the ini file. Load the default CXI protocol if None.
        datatypes : dict, optional
            Dictionary with attributes' datatypes. 'float', 'int', or 'bool'
            are allowed. Initialized with `ini_file` if None.
        is_data : dict, optional
            Dictionary with the flags if the attribute is of data type.
            Data type is 2- or 3-dimensional and has the same data shape
            as `data`. Initialized with `ini_file` if None.
        default_paths : dict, optional
            Dictionary with attributes path in the CXI file. Initialized with
            `ini_file` if None.

        Returns
        -------
        CXIProtocol
            A :class:`CXIProtocol` object with all the attributes imported
            from the ini file.

        See Also
        --------
        cxi_protocol : more details about the default CXI protocol.
        """
        kwargs = cls._import_ini(ini_file)
        if not datatypes is None:
            kwargs['datatypes'].update(**datatypes)
        if not default_paths is None:
            kwargs['default_paths'].update(**default_paths)
        if not is_data is None:
            kwargs['is_data'].update(**is_data)
        return cls(datatypes=kwargs['datatypes'], default_paths=kwargs['default_paths'],
                   is_data=kwargs['is_data'])

    def parser_from_template(self, path):
        """Return a :class:`configparser.ConfigParser` object using
        an ini file template.

        Parameters
        ----------
        path : str
            Path to the ini file template.

        Returns
        -------
        configparser.ConfigParser
            Parser object with the attributes populated according
            to the protocol.
        """
        ini_template = configparser.ConfigParser()
        ini_template.read(path)
        parser = configparser.ConfigParser()
        for section in ini_template:
            parser[section] = {option: ini_template[section][option].format(**self.default_paths)
                               for option in ini_template[section]}
        return parser

    def __iter__(self):
        return self.default_paths.__iter__()

    def __contains__(self, attr):
        return attr in self.default_paths

    def get_default_path(self, attr, value=None):
        """Return the atrribute's default path in the CXI file.
        Return `value` if `attr` is not found.

        Parameters
        ----------
        attr : str
            The attribute to look for.

        value : str, optional
            value which is returned if the `attr` is not found.

        Returns
        -------
        str or None
            Attribute's cxi file path.
        """
        return self.default_paths.get(attr, value)

    def get_dtype(self, attr, value=None):
        """Return the attribute's data type.
        Return `value` if `attr` is not found.

        Parameters
        ----------
        attr : str
            The data attribute.
        value : str, optional
            value which is returned if the `attr` is not found.

        Returns
        -------
        type or None
            Attribute's data type.
        """
        return self.known_types.get(self.datatypes.get(attr), value)

    def get_is_data(self, attr, value=False):
        """Return if the attribute is of data type. Data type is
        2- or 3-dimensional and has the same data shape as `data`.

        Parameters
        ----------
        attr : str
            The data attribute.
        value : str, optional
            value which is returned if the `attr` is not found.

        Returns
        -------
        bool
            True if `attr` is of data type.
        """
        is_data = self.is_data.get(attr, value)
        if isinstance(is_data, str):
            return is_data in ['True', 'true', '1', 'y', 'yes']
        else:
            return bool(is_data)

    @staticmethod
    def get_dataset_shapes(cxi_path, cxi_obj):
        """Visit recursively all the underlying datasets and return
        their names and shapes.

        Parameters
        ----------
        cxi_path : str
            Path of the HDF5 group.
        cxi_obj: h5py.Group
            Group object.

        Returns
        -------
        shapes : list
            List of all the datasets and their shapes inside `cxi_obj`.
        """
        shapes = []

        def caller(sub_path, obj):
            if isinstance(obj, h5py.Dataset):
                shapes.append((os.path.join(cxi_path, sub_path), obj.shape))

        cxi_obj.visititems(caller)
        return shapes

    def read_shape(self, cxi_file, cxi_path):
        """Read data shapes from the CXI file `cxi_file` at
        the `cxi_path` path inside the CXI file recursively.

        Parameters
        ----------
        cxi_file : h5py.File
            h5py File object of the CXI file.
        cxi_path : str, optional
            Path to the data attribute.

        Returns
        -------
        list of tuples or None
            The attribute's data shapes extracted from the CXI file.
            Returns None if no datasets has been found.
        """
        if not cxi_path is None and cxi_path in cxi_file:
            cxi_obj = cxi_file[cxi_path]
            if isinstance(cxi_obj, h5py.Dataset):
                return [(cxi_path, cxi_obj.shape),]
            elif isinstance(cxi_obj, h5py.Group):
                return self.get_dataset_shapes(cxi_path, cxi_obj)
            else:
                raise ValueError(f"Invalid CXI object at '{cxi_path:s}'")

        return None

    def read_cxi(self, attr, cxi_file, cxi_path=None):
        """Read `attr` from the CXI file `cxi_file` at the `cxi_path`
        path. If `cxi_path` is not provided the default path for the
        given attribute is used.

        Parameters
        ----------
        attr : str
            Data attribute.
        cxi_file : h5py.File
            h5py File object of the CXI file.
        cxi_path : str, optional
            Path to the data attribute. If `cxi_path` is None,
            the path will be inferred according to the protocol.

        Returns
        -------
        numpy.ndarray or None
            The value of the attribute extracted from the CXI file.
        """
        if cxi_path is None:
            cxi_path = self.get_default_path(attr)
        shapes = self.read_shape(cxi_file, cxi_path)
        if shapes is None:
            return None

        data_dict = {}
        prefix = os.path.commonpath(list([cxi_path for cxi_path, _ in shapes]))
        for cxi_path, _ in shapes:
            key = os.path.relpath(cxi_path, prefix)
            if key.isnumeric():
                key = int(key)
            data_dict[key] = cxi_file[cxi_path][()]

        if '.' in data_dict:
            data = np.asarray(data_dict['.'], dtype=self.get_dtype(attr))
            if data.size == 1:
                return np.asscalar(data)
            return data

        if data_dict:
            return data_dict

        return None

    @staticmethod
    def _write_dset(cxi_file, cxi_path, data, dtype, **kwargs):
        try:
            cxi_file[cxi_path][...] = data
        except TypeError:
            del cxi_file[cxi_path]
            cxi_file.create_dataset(cxi_path, data=data, dtype=dtype, **kwargs)
        except KeyError:
            cxi_file.create_dataset(cxi_path, data=data, dtype=dtype, **kwargs)

    def write_cxi(self, attr, data, cxi_file, cxi_path=None):
        """Write data to the CXI file `cxi_file` under the path
        specified by the protocol. If `cxi_path` or `dtype` argument
        are provided, it will override the protocol.

        Parameters
        ----------
        attr : str
            Data attribute.
        data : numpy.ndarray
            Data which is bound to be saved.
        cxi_file : h5py.File
            :class:`h5py.File` object of the CXI file.
        cxi_path : str, optional
            Path to the data attribute. If `cxi_path` is None,
            the path will be inferred according to the protocol.

        Raises
        ------
        ValueError
            If `overwrite` is False and the data is already present
            at the given location in `cxi_file`.
        """
        if data is None:
            pass
        else:
            if cxi_path is None:
                cxi_path = self.get_default_path(attr, cxi_path)

            if isinstance(data, dict):
                for key, val in data.items():
                    self._write_dset(cxi_file, os.path.join(cxi_path, str(key)), data=val,
                                     dtype=self.get_dtype(attr))

            elif self.get_is_data(attr):
                self._write_dset(cxi_file, cxi_path, data=data, dtype=self.get_dtype(attr),
                                 chunks=(1,) + data.shape[1:], maxshape=(None,) + data.shape[1:])

            else:
                self._write_dset(cxi_file, cxi_path, data=data, dtype=self.get_dtype(attr))

class CXILoader(CXIProtocol):
    """CXI file loader class. Loads data from a
    CXI file and returns a :class:`CrystData` container or a
    :class:`dict` with the data. Search data in the paths
    provided by `protocol` and `load_paths`.

    Parameters
    ----------
    protocol : CXIProtocol, optional
        Protocol object. The default protocol is used if None.
        Default protocol is used if not provided.
    load_paths : dict, optional
        Extra paths to the data attributes in a CXI file,
        which override `protocol`. Accepts only the attributes
        enlisted in `protocol`. Default paths are used if
        not provided.
    policy : dict, optional
        A dictionary with loading policy. Contains all the
        attributes that are available in `protocol` with their
        corresponding flags. If a flag is True, the attribute
        will be loaded from a file. Default policy is used if
        not provided.

    Attributes
    ----------
    datatypes : dict
        Dictionary with attributes' datatypes. 'float', 'int', or 'bool'
        are allowed.
    default_paths : dict
        Dictionary with attributes' CXI default file paths.
    is_data : dict
        Dictionary with the flags if the attribute is of data type.
        Data type is 2- or 3-dimensional and has the same data shape
        as `data`.
    load_paths : dict
        Extra set of paths to the attributes enlisted in `datatypes`.
    policy: dict
        Loading policy.

    See Also
    --------
    cxi_protocol : Full list of data attributes and configuration
        parameters.
    CrystData : Data container with all the data  necessary for
        Speckle Tracking.
    """
    attr_dict = {'datatypes': ('ALL', ), 'default_paths': ('ALL', ),
                 'load_paths': ('ALL',), 'is_data': ('ALL', ), 'policy': ('ALL', )}
    fmt_dict = {'datatypes': 'str', 'default_paths': 'str',
                'load_paths': 'str', 'is_data': 'str', 'policy': 'str'}

    def __init__(self, protocol=None, load_paths=None, policy=None):
        if protocol is None:
            protocol = CXIProtocol()
        if load_paths is None:
            load_paths = self._import_ini(CXI_PROTOCOL)['load_paths']
        if policy is None:
            policy = self._import_ini(CXI_PROTOCOL)['policy']

        load_paths = {attr: self.str_to_list(paths) for attr, paths in load_paths.items()
                      if attr in protocol}
        policy = {attr: flag for attr, flag in policy.items() if attr in protocol}

        super(CXIProtocol, self).__init__(datatypes=protocol.datatypes, policy=policy,
                                          default_paths=protocol.default_paths,
                                          is_data=protocol.is_data, load_paths=load_paths)

    @staticmethod
    def str_to_list(strings):
        """Convert `strings` to a list of strings.

        Parameters
        ----------
        strings : str or list
            String or a list of strings

        Returns
        -------
        list
            List of strings.
        """
        if isinstance(strings, (str, list)):
            if isinstance(strings, str):
                return [strings,]
            return strings

        raise ValueError('strings must be a string or a list of strings')

    @classmethod
    def import_default(cls, protocol=None, load_paths=None, policy=None):
        """Return the default :class:`CXILoader` object. Extra arguments
        override the default values if provided.

        Parameters
        ----------
        protocol : CXIProtocol, optional
            Protocol object.
        load_paths : dict, optional
            Extra paths to the data attributes in a CXI file,
            which override `protocol`. Accepts only the attributes
            enlisted in `protocol`.
        policy : dict, optional
            A dictionary with loading policy. Contains all the
            attributes that are available in `protocol` and the
            corresponding flags. If a flag is True, the attribute
            will be loaded from a file.

        Returns
        -------
        CXILoader
            A :class:`CXILoader` object with the default parameters.

        See Also
        --------
        cxi_protocol : more details about the default CXI loader.
        """
        return cls.import_ini(CXI_PROTOCOL, protocol, load_paths, policy)

    @classmethod
    def import_ini(cls, ini_file, protocol=None, load_paths=None, policy=None):
        """Initialize a :class:`CXILoader` object class with an
        ini file.

        Parameters
        ----------
        ini_file : str
            Path to the ini file. Loads the default CXI loader if None.
        protocol : CXIProtocol, optional
            Protocol object. Initialized with `ini_file` if None.
        load_paths : dict, optional
            Extra paths to the data attributes in a CXI file,
            which override `protocol`. Accepts only the attributes
            enlisted in `protocol`. Initialized with `ini_file`
            if None.
        policy : dict, optional
            A dictionary with loading policy. Contains all the
            attributes that are available in `protocol` and the
            corresponding flags. If a flag is True, the attribute
            will be loaded from a file. Initialized with `ini_file`
            if None.

        Returns
        -------
        CXILoader
            A :class:`CXILoader` object with all the attributes imported
            from the ini file.

        See Also
        --------
        cxi_protocol : more details about the default CXI loader.
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

    def get_load_paths(self, attr, value=None):
        """Return the atrribute's path in the cxi file.
        Return `value` if `attr` is not found.

        Parameters
        ----------
        attr : str
            The attribute to look for.
        value : str, optional
            value which is returned if the `attr` is not found.

        Returns
        -------
        list
            Set of attribute's paths.
        """
        paths = [super(CXILoader, self).get_default_path(attr, value)]
        if attr in self.load_paths:
            paths.extend(self.load_paths[attr])
        return paths

    def get_policy(self, attr, value=False):
        """Return the atrribute's loding policy.

        Parameters
        ----------
        attr : str
            The attribute to look for.
        value : str, optional
            value which is returned if the `attr` is not found.

        Returns
        -------
        bool
            Attributes' loding policy.
        """
        policy = self.policy.get(attr, value)
        if isinstance(policy, str):
            return policy in ['True', 'true', '1', 'y', 'yes']
        else:
            return bool(policy)

    def get_protocol(self):
        """Return a CXI protocol from the loader.

        Returns
        -------
        CXIProtocol
            CXI protocol.
        """
        return CXIProtocol(datatypes=self.datatypes,
                           default_paths=self.default_paths,
                           is_data=self.is_data)

    def find_path(self, attr, cxi_file):
        """Find attribute's path in a CXI file `cxi_file`.

        Parameters
        ----------
        attr : str
            Data attribute.
        cxi_file : h5py.File
            :class:`h5py.File` object of the CXI file.

        Returns
        -------
        str or None
            Atrribute's path in the CXI file,
            return None if the attribute is not found.
        """
        paths = self.get_load_paths(attr)
        for path in paths:
            if path in cxi_file:
                return path
        return None

    def load_attributes(self, master_file):
        """Return attributes' values from a CXI file at
        the given `master_file`.

        Parameters
        ----------
        master_file : str
            Path to the master CXI file.

        Returns
        -------
        attr_dict : dict
            Dictionary with the attributes retrieved from
            the CXI file.
        """
        if not isinstance(master_file, str):
            raise ValueError('master_file must be a string')

        attr_dict = {}
        with h5py.File(master_file, 'r') as cxi_file:
            for attr in self:
                if not self.get_is_data(attr):
                    if self.get_policy(attr, False):
                        cxi_path = self.find_path(attr, cxi_file)
                        attr_dict[attr] = self.read_cxi(attr, cxi_file, cxi_path)
                    else:
                        attr_dict[attr] = None
        return attr_dict

    def read_indices(self, attr, data_files):
        """Retrieve the indices of the datasets from the CXI files for the
        given attribute `attr`.

        Parameters
        ----------
        attr : str
            The attribute to read.
        data_files : str or list of str
            Paths to the data CXI files.

        Returns
        -------
        paths : np.ndarray or None.
            List of the file paths. None if no data is found.
        cxi_paths : np.ndarray or None
            List of the paths inside the files. None if no data
            is found.
        indices : np.ndarray or None
            List of the frame indices. None if no data is found.
        """
        data_files = self.str_to_list(data_files)

        paths, cxi_paths, indices = [], [], []
        for path in data_files:
            with h5py.File(path, 'r') as cxi_file:
                shape = self.read_shape(cxi_file, self.find_path(attr, cxi_file))
            if shape is not None:
                for cxi_path, dset_shape in shape:
                    if len(dset_shape) == 3:
                        paths.append(np.repeat(path, dset_shape[0]))
                        cxi_paths.append(np.repeat(cxi_path, dset_shape[0]))
                        indices.append(np.arange(dset_shape[0]))
                    elif len(dset_shape) == 2:
                        paths.append(np.atleast_1d(path))
                        cxi_paths.append(np.atleast_1d(cxi_path))
                        indices.append(np.atleast_1d(slice(None)))
                    else:
                        raise ValueError('Dataset must be 2- or 3-dimensional')

        if len(paths) == 0:
            return np.asarray(paths), np.asarray(cxi_paths), np.asarray(indices)
        elif len(paths) == 1:
            return paths[0], cxi_paths[0], indices[0]
        else:
            return (np.concatenate(paths), np.concatenate(cxi_paths),
                    np.concatenate(indices))

    def load_data(self, attr, paths, cxi_paths, indices, verbose=True):
        """Retrieve the data for the given attribute `attr` from the
        CXI files. Uses the result from :method:`CXILoader.read_indices`
        method.

        Parameters
        ----------
        attr : str
            The attribute to read.
        paths : np.ndarray or None.
            List of the file paths.
        cxi_paths : np.ndarray or None
            List of the paths inside the files.
        indices : np.ndarray or None
            List of the frame indices.
        verbose : bool, optional
            Print the progress bar if True.

        Returns
        -------
        data : np.ndarray
            Data array retrieved from the CXI files.
        """
        data = []
        for path, cxi_path, index in tqdm(zip(paths, cxi_paths, indices),
                                          disable=not verbose, total=paths.size,
                                          desc=f'Loading {attr:s}'):
            with h5py.File(path, 'r') as cxi_file:
                data.append(cxi_file[cxi_path][index])

        return np.asarray(np.stack(data, axis=0), dtype=self.get_dtype(attr))

    def load_to_dict(self, data_files, master_file=None, frame_indices=None, **attributes):
        """Load data from the CXI files and return a :class:`dict` with
        all the data fetched from the `data_files` and `master_file`.

        Parameters
        ----------
        data_files : str or list of str
            Paths to the data CXI files.
        master_file : str, optional
            Path to the master CXI file. First file in `data_files`
            if not provided.
        frame_indices : sequence of int, optional
            Array of frame indices to load. Loads all the frames by
            default.
        **attributes : dict, optional
            Dictionary of attribute values, that override the loaded
            values.

        Returns
        -------
        dict
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
            frame_indices = np.arange(self.read_indices('data', data_files)[0].size)

        for attr in self:
            if self.get_is_data(attr):
                paths, cxi_paths, indices = self.read_indices(attr, data_files)
                if paths.size > 0:
                    good_frames = frame_indices[frame_indices < paths.size]
                    data_dict[attr] = self.load_data(attr, paths[good_frames],
                                                     cxi_paths[good_frames],
                                                     indices[good_frames])

        if data_dict['frames'] is None:
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

        return data_dict

    def load(self, data_files, master_file=None, frame_indices=None, **attributes):
        """Load data from the CXI files and return a :class:`CrystData` container
        with all the data fetched from the `data_files` and `master_file`.

        Parameters
        ----------
        data_files : str or list of str
            Paths to the data CXI files.
        master_file : str, optional
            Path to the master CXI file. First file in `data_files`
            if not provided.
        frame_indices : sequence of int, optional
            Array of frame indices to load. Loads all the frames by
            default.
        **attributes : dict
            Dictionary of attribute values,
            which will be parsed to the `CrystData` object instead.

        Returns
        -------
        CrystData
            Data container object with all the necessary data
            for the Speckle Tracking algorithm.
        """
        return CrystData(self.get_protocol(), **self.load_to_dict(data_files, master_file,
                                                                  frame_indices, **attributes))
