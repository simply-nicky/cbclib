"""Examples
--------
Generate the default built-in log protocol:

>>> import cbclib as cbc
>>> cbc.LogProtocol()
{'datatypes': {'exposure': 'float', 'n_points': 'int', 'n_steps': 'int', '...':
'...'}, 'log_keys': {'exposure': ['Exposure'], 'n_points': ['Points count'],
'n_steps': ['Steps count'], '...': '...'}, 'part_keys': {'exposure': 'Type: Method',
'n_points': 'Type: Scan', 'n_steps': 'Type: Scan', '...': '...'}}
"""
import os
import re
import numpy as np
from .ini_parser import ROOT_PATH, INIParser
from .cxi_protocol import CXILoader, CXIProtocol
from .data_processing import CrystData

LOG_PROTOCOL = os.path.join(ROOT_PATH, 'config/log_protocol.ini')

class LogProtocol(INIParser):
    """Log file protocol class. Contains log file keys to retrieve
    and the data types of the corresponding values.

    Parameters
    ----------
    datatypes : dict, optional
        Dictionary with attributes' datatypes. 'float', 'int', 'bool',
        or 'str' are allowed.
    log_keys : dict, optional
        Dictionary with attributes' log file keys.

    Attributes
    ----------
    datatypes : dict
        Dictionary with attributes' datatypes. 'float', 'int', 'bool',
        or 'str' are allowed.
    log_keys : dict
        Dictionary with attributes' log file keys.

    See Also
    --------
    protocol : Full list of data attributes and configuration
        parameters.
    """
    attr_dict = {'datatypes': ('ALL',), 'log_keys': ('ALL',), 'part_keys': ('ALL',)}
    fmt_dict = {'datatypes': 'str', 'log_keys': 'str', 'part_keys': 'str'}
    unit_dict = {'percent': 1e-2, 'mm': 1e-3, 'mdeg': 1.7453292519943296e-05, 'µm,um': 1e-6,
                 'udeg,µdeg': 1.7453292519943296e-08, 'nm': 1e-9, 'ndeg': 1.7453292519943296e-11,
                 'pm': 1e-12,  'pdeg': 1.7453292519943296e-14}

    def __init__(self, datatypes=None, log_keys=None, part_keys=None):
        if datatypes is None:
            datatypes = self._import_ini(LOG_PROTOCOL)['datatypes']
        if log_keys is None:
            log_keys = self._import_ini(LOG_PROTOCOL)['log_keys']
        if part_keys is None:
            part_keys = self._import_ini(LOG_PROTOCOL)['part_keys']
        log_keys = {attr: val for attr, val in log_keys.items() if attr in datatypes}
        datatypes = {attr: val for attr, val in datatypes.items() if attr in log_keys}
        super(LogProtocol, self).__init__(datatypes=datatypes, log_keys=log_keys,
                                          part_keys=part_keys)

    @classmethod
    def import_default(cls, datatypes=None, log_keys=None, part_keys=None):
        """Return the default :class:`LogProtocol` object. Extra arguments
        override the default values if provided.

        Parameters
        ----------
        datatypes : dict, optional
            Dictionary with attributes' datatypes. 'float', 'int', or 'bool'
            are allowed. Initialized with `ini_file` if None.
        log_keys : dict, optional
            Dictionary with attributes' log file keys. Initialized with
            `ini_file` if None.

        Returns
        -------
        LogProtocol
            A :class:`LogProtocol` object with the default parameters.

        See Also
        --------
        log_protocol : more details about the default CXI protocol.
        """
        return cls.import_ini(LOG_PROTOCOL, datatypes, log_keys, part_keys)

    @classmethod
    def import_ini(cls, ini_file, datatypes=None, log_keys=None, part_keys=None):
        """Initialize a :class:`LogProtocol` object class with an
        ini file.

        Parameters
        ----------
        ini_file : str
            Path to the ini file. Load the default log protocol if None.
        datatypes : dict, optional
            Dictionary with attributes' datatypes. 'float', 'int', or 'bool'
            are allowed. Initialized with `ini_file` if None.
        log_keys : dict, optional
            Dictionary with attributes' log file keys. Initialized with
            `ini_file` if None.

        Returns
        -------
        LogProtocol
            A :class:`LogProtocol` object with all the attributes imported
            from the ini file.
        """
        kwargs = cls._import_ini(ini_file)
        if not datatypes is None:
            kwargs['datatypes'].update(**datatypes)
        if not log_keys is None:
            kwargs['log_keys'].update(**log_keys)
        if not part_keys is None:
            kwargs['part_keys'].update(**part_keys)
        return cls(datatypes=kwargs['datatypes'], log_keys=kwargs['log_keys'],
                   part_keys=kwargs['part_keys'])

    @classmethod
    def _get_unit(cls, key):
        for unit_key in cls.unit_dict:
            units = unit_key.split(',')
            for unit in units:
                if unit in key:
                    return cls.unit_dict[unit_key]
        return 1.

    @classmethod
    def _has_unit(cls, key):
        has_unit = False
        for unit_key in cls.unit_dict:
            units = unit_key.split(',')
            for unit in units:
                has_unit |= (unit in key)
        return has_unit

    def load_attributes(self, path):
        """Return attributes' values from a log file at
        the given `path`.

        Parameters
        ----------
        path : str
            Path to the log file.

        Returns
        -------
        attr_dict : dict
            Dictionary with the attributes retrieved from
            the log file.
        """
        if not isinstance(path, str):
            raise ValueError('path must be a string')
        with open(path, 'r') as log_file:
            log_str = ''
            for line in log_file:
                if line.startswith('# '):
                    log_str += line.strip('# ')
                else:
                    break

        # List all the sector names
        part_keys = list(self.part_keys.values())

        # Divide log into sectors
        parts_list = [part for part in re.split('(' + '|'.join(part_keys) + \
                      '|--------------------------------)\n*', log_str) if part]


        # Rearange sectors into a dictionary
        parts = {}
        for idx, part in enumerate(parts_list):
            if part in part_keys:
                if part == 'Session logged attributes':
                    attr_keys, attr_vals = parts_list[idx + 1].strip('\n').split('\n')
                    parts['Session logged attributes'] = ''
                    for key, val in zip(attr_keys.split(';'), attr_vals.split(';')):
                        parts['Session logged attributes'] += key + ': ' + val + '\n'
                else:
                    val = parts_list[idx + 1]
                    match = re.search(r'Device:.*\n', val)
                    if match:
                        name = match[0].split(': ')[-1][:-1]
                        parts[part + ', ' + name] = val

        # Populate attributes dictionary
        attr_dict = {part_name: {} for part_name in parts}
        for part_name, part in parts.items():
            for attr, part_key in self.part_keys.items():
                if part_key in part_name:
                    for log_key in self.log_keys[attr]:
                        # Find the attribute's mention and divide it into a key and value pair
                        match = re.search(log_key + r'.*\n', part)
                        if match:
                            raw_str = match[0]
                            raw_val = raw_str.strip('\n').split(': ')[1]
                            # Extract a number string
                            val_num = re.search(r'[-]*\d+[.]*\d*', raw_val)
                            dtype = self.known_types[self.datatypes[attr]]
                            attr_dict[part_name][attr] = dtype(val_num[0] if val_num else raw_val)
                            # Apply unit conversion if needed
                            if np.issubdtype(dtype, np.floating):
                                attr_dict[part_name][attr] *= self._get_unit(raw_str)
        return attr_dict

    def load_data(self, path, frame_indices=None):
        """Retrieve the main data array from the log file.

        Parameters
        ----------
        path : str
            Path to the log file.

        Returns
        -------
        data : dict
            Dictionary with data fields and their names retrieved
            from the log file.
        """
        with open(path, 'r') as log_file:
            for line in log_file:
                if line.startswith('# '):
                    keys_line = line.strip('# ')
                else:
                    data_line = line
                    break

        keys = keys_line.strip('\n').split(';')
        data_strings = data_line.strip('\n').split(';')

        dtypes = {'names': [], 'formats': []}
        converters = {}
        for idx, (key, val) in enumerate(zip(keys, data_strings)):
            dtypes['names'].append(key)
            unit = self._get_unit(key)
            if 'float' in key:
                dtypes['formats'].append(np.float_)
                converters[idx] = lambda item, unit=unit: unit * float(item)
            elif 'int' in key:
                if self._has_unit(key):
                    converters[idx] = lambda item, unit=unit: unit * float(item)
                    dtypes['formats'].append(np.float_)
                else:
                    dtypes['formats'].append(np.int)
            elif 'Array' in key:
                dtypes['formats'].append(np.ndarray)
                converters[idx] = lambda item, unit=unit: np.array([float(part.strip(b' []')) * unit
                                                                    for part in item.split(b',')])
            else:
                dtypes['formats'].append('<S' + str(len(val)))
                converters[idx] = lambda item: item.strip(b' []')
        if frame_indices is None:
            return dict(zip(keys, np.loadtxt(path, delimiter=';',
                                             converters=converters,
                                             dtype=dtypes, unpack=True)))
        skiprows = np.min(frame_indices)
        max_rows = np.max(frame_indices - skiprows)
        data_tuple = np.loadtxt(path, delimiter=';', converters=converters,
                                dtype=dtypes, unpack=True, skiprows=skiprows,
                                max_rows=max_rows + 1)
        return {key: data[frame_indices - skiprows] for key, data in zip(keys, data_tuple)}

def converter_petra(dir_path, scan_num, frame_indices=None, **attributes):
    cxi_loader = CXILoader()
    log_prt = LogProtocol()

    h5_dir = os.path.join(dir_path, f'scan_frames/Scan_{scan_num:d}')
    log_path = os.path.join(dir_path, f'server_log/Scan_logs/Scan_{scan_num:d}.log')
    h5_files = sorted([os.path.join(h5_dir, path) for path in os.listdir(h5_dir)
                       if path.endswith(('LambdaFar.nxs', '.h5')) and not path.endswith('master.h5')])
    h5_master = [os.path.join(h5_dir, path) for path in os.listdir(h5_dir)
                 if path.endswith('master.h5')]
    if h5_master:
        h5_master = h5_master[0]
    else:
        h5_master = h5_files[0]

    log_attrs = log_prt.load_attributes(log_path)
    log_data = log_prt.load_data(log_path, frame_indices)

    data_dict = cxi_loader.load_to_dict(h5_files, h5_master, frame_indices, **attributes)

    x_sample = log_attrs['Session logged attributes'].get('x_sample', 0.0)
    y_sample = log_attrs['Session logged attributes'].get('y_sample', 0.0)
    z_sample = log_attrs['Session logged attributes'].get('z_sample', 0.0)
    r_sample = log_attrs['Session logged attributes'].get('r_sample', 0.0)
    data_dict['translations'] = np.tile([[x_sample, y_sample, z_sample]],
                                        (data_dict['data'].shape[0], 1))
    data_dict['tilts'] = r_sample * np.ones(data_dict['data'].shape[0])
    for data_key, log_dset in log_data.items():
        for log_key in log_prt.log_keys['x_sample']:
            if log_key in data_key:
                data_dict['translations'][:, 0] = log_dset
        for log_key in log_prt.log_keys['y_sample']:
            if log_key in data_key:
                data_dict['translations'][:, 1] = log_dset
        for log_key in log_prt.log_keys['z_sample']:
            if log_key in data_key:
                data_dict['translations'][:, 2] = log_dset
        for log_key in log_prt.log_keys['r_sample']:
            if log_key in data_key:
                data_dict['tilts'] = log_dset

    return CrystData(protocol=CXIProtocol(), **data_dict)
