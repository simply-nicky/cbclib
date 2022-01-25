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
from __future__ import annotations
import os
import re
from typing import Any, Dict, Iterable, List, Optional
import numpy as np
from .ini_parser import ROOT_PATH, INIParser
from .cxi_protocol import CXIStore
from .data_processing import CrystData

LOG_PROTOCOL = os.path.join(ROOT_PATH, 'config/log_protocol.ini')

class LogProtocol(INIParser):
    """Log file protocol class. Contains log file keys to retrieve
    and the data types of the corresponding values.

    Attributes:
        datatypes : Dictionary with attributes' datatypes. 'float', 'int',
            'bool', or 'str' are allowed.
        log_keys : Dictionary with attributes' log file keys.
        part_keys : Dictionary with the part names inside the log file
            where the attributes are stored.
    """
    attr_dict = {'datatypes': ('ALL',), 'log_keys': ('ALL',), 'part_keys': ('ALL',)}
    fmt_dict = {'datatypes': 'str', 'log_keys': 'str', 'part_keys': 'str'}
    unit_dict = {'percent': 1e-2, 'mm,mdeg': 1e-3, 'µm,um,udeg,µdeg': 1e-6,
                 'nm,ndeg': 1e-9, 'pm,pdeg': 1e-12}

    def __init__(self, datatypes: Dict[str, str], log_keys: Dict[str, List[str]],
                 part_keys: Dict[str, str]) -> None:
        """
        Args:
            datatypes : Dictionary with attributes' datatypes. 'float', 'int',
                'bool', or 'str' are allowed.
            log_keys : Dictionary with attributes' log file keys.
            part_keys : Dictionary with the part names inside the log file
                where the attributes are stored.
        """
        log_keys = {attr: val for attr, val in log_keys.items() if attr in datatypes}
        datatypes = {attr: val for attr, val in datatypes.items() if attr in log_keys}
        super(LogProtocol, self).__init__(datatypes=datatypes, log_keys=log_keys,
                                          part_keys=part_keys)

    @classmethod
    def import_default(cls, datatypes: Optional[Dict[str, str]]=None,
                       log_keys: Optional[Dict[str, List[str]]]=None,
                       part_keys: Optional[Dict[str, str]]=None) -> LogProtocol:
        """Return the default :class:`LogProtocol` object. Extra arguments
        override the default values if provided.

        Args:
            datatypes : Dictionary with attributes' datatypes. 'float', 'int',
                or 'bool' are allowed.
            log_keys : Dictionary with attributes' log file keys.
            part_keys : Dictionary with the part names inside the log file
                where the attributes are stored.

        Returns:
            A :class:`LogProtocol` object with the default parameters.
        """
        return cls.import_ini(LOG_PROTOCOL, datatypes, log_keys, part_keys)

    @classmethod
    def import_ini(cls, ini_file: str, datatypes: Optional[Dict[str, str]]=None,
                   log_keys: Optional[Dict[str, List[str]]]=None,
                   part_keys: Optional[Dict[str, str]]=None) -> LogProtocol:
        """Initialize a :class:`LogProtocol` object class with an
        ini file.

        Args:
            ini_file : Path to the ini file. Load the default log protocol if None.
            datatypes : Dictionary with attributes' datatypes. 'float', 'int',
                or 'bool' are allowed. Initialized with `ini_file` if None.
            log_keys : Dictionary with attributes' log file keys. Initialized with
                `ini_file` if None.
            part_keys : Dictionary with the part names inside the log file
                where the attributes are stored. Initialized with `ini_file`
                if None.

        Returns:
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
    def _get_unit(cls, key: str) -> float:
        for unit_key in cls.unit_dict:
            units = unit_key.split(',')
            for unit in units:
                if unit in key:
                    return cls.unit_dict[unit_key]
        return 1.0

    @classmethod
    def _has_unit(cls, key: str) -> bool:
        has_unit = False
        for unit_key in cls.unit_dict:
            units = unit_key.split(',')
            for unit in units:
                has_unit |= (unit in key)
        return has_unit

    def load_attributes(self, path: str) -> Dict[str, Any]:
        """Return attributes' values from a log file at
        the given `path`.

        Args:
            path : Path to the log file.

        Returns:
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

    def load_data(self, path: str, frame_indices: Optional[Iterable[int]]=None) -> Dict[str, np.ndarray]:
        """Retrieve the main data array from the log file.

        Args:
            path : Path to the log file.
            frame_indices : Array of data indices to load. Loads info for all
                the frames by default.

        Returns:
            Dictionary with data fields and their names retrieved
            from the log file.
        """
        if frame_indices is not None:
            frame_indices.sort()

        row_cnt = 0
        with open(path, 'r') as log_file:
            for line_idx, line in enumerate(log_file):
                if line.startswith('# '):
                    if 'WARNING' not in line:
                        keys_line = line.strip('# ')
                else:
                    data_line = line

                    if row_cnt == 0:
                        first_row = line_idx
                    if frame_indices is not None and row_cnt == frame_indices[0]:
                        skiprows = line_idx - first_row
                    if frame_indices is not None and row_cnt == frame_indices[-1]:
                        max_rows = line_idx - skiprows
                        break

                    row_cnt += 1
            else:
                if frame_indices is None:
                    frame_indices = np.arange(row_cnt)
                    skiprows = 0
                    max_rows = line_idx - skiprows
                else:
                    frame_indices = frame_indices[:np.searchsorted(frame_indices, row_cnt)]
                    if not frame_indices.size:
                        skiprows = line_idx
                    max_rows = line_idx - skiprows

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

        data_tuple = np.loadtxt(path, delimiter=';', converters=converters,
                                dtype=dtypes, unpack=True, skiprows=skiprows,
                                max_rows=max_rows + 1)
        data_dict = {key: data[frame_indices - skiprows] for key, data in zip(keys, data_tuple)}
        data_dict['indices'] = frame_indices
        return data_dict

def converter_petra(dir_path, scan_num, out_path, **kwargs):
    log_prt = LogProtocol.import_default()

    h5_dir = os.path.join(dir_path, f'scan_frames/Scan_{scan_num:d}')
    log_path = os.path.join(dir_path, f'server_log/Scan_logs/Scan_{scan_num:d}.log')
    h5_files = sorted([os.path.join(h5_dir, path) for path in os.listdir(h5_dir)
                       if path.endswith(('LambdaFar.nxs', '.h5'))])

    files = CXIStore(input_files=h5_files, output_file=out_path)
    n_steps = files.indices().size

    log_attrs = log_prt.load_attributes(log_path)
    log_data = log_prt.load_data(log_path)

    x_sample = log_attrs['Session logged attributes'].get('x_sample', 0.0)
    y_sample = log_attrs['Session logged attributes'].get('y_sample', 0.0)
    z_sample = log_attrs['Session logged attributes'].get('z_sample', 0.0)
    r_sample = log_attrs['Session logged attributes'].get('r_sample', 0.0)
    translations = np.tile([[x_sample, y_sample, z_sample]], (n_steps, 1))
    tilts = r_sample * np.ones(n_steps)
    for data_key, log_dset in log_data.items():
        for log_key in log_prt.log_keys['x_sample']:
            if log_key in data_key:
                translations[:, 0] = log_dset[:n_steps]
        for log_key in log_prt.log_keys['y_sample']:
            if log_key in data_key:
                translations[:, 1] = log_dset[:n_steps]
        for log_key in log_prt.log_keys['z_sample']:
            if log_key in data_key:
                translations[:, 2] = log_dset[:n_steps]
        for log_key in log_prt.log_keys['r_sample']:
            if log_key in data_key:
                tilts = log_dset[:n_steps]

    return CrystData(files=files, translations=translations, tilts=tilts, **kwargs)
