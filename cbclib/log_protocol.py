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
from dataclasses import dataclass
import os
import re
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Tuple
import numpy as np
from .data_container import INIContainer
from .cxi_protocol import CXIStore
from .data_processing import CrystData

LOG_PROTOCOL = os.path.join(os.path.dirname(__file__), 'config/log_protocol.ini')

@dataclass
class LogProtocol(INIContainer):
    """Log file protocol class. Contains log file keys to retrieve
    and the data types of the corresponding values.

    Attributes:
        datatypes : Dictionary with attributes' datatypes. 'float', 'int',
            'bool', or 'str' are allowed.
        log_keys : Dictionary with attributes' log file keys.
        part_keys : Dictionary with the part names inside the log file
            where the attributes are stored.
    """
    __ini_fields__ = {'datatypes': 'datatypes', 'log_keys': 'log_keys', 'part_keys': 'part_keys'}

    datatypes : Dict[str, str]
    log_keys : Dict[str, List[str]]
    part_keys : Dict[str, str]

    known_types: ClassVar[Dict[str, Any]] = {'int': int, 'bool': bool, 'float': float, 'str': str}
    unit_dict: ClassVar[Dict[str, float]] = {'percent': 1e-2, 'mm,mdeg': 1e-3,
                                             'µm,um,udeg,µdeg': 1e-6,
                                             'nm,ndeg': 1e-9, 'pm,pdeg': 1e-12}

    def __post_init__(self):
        """
        Args:
            datatypes : Dictionary with attributes' datatypes. 'float', 'int',
                'bool', or 'str' are allowed.
            log_keys : Dictionary with attributes' log file keys.
            part_keys : Dictionary with the part names inside the log file
                where the attributes are stored.
        """
        self.log_keys = {attr: self.str_to_list(val) for attr, val in self.log_keys.items() if attr in self.datatypes}
        self.part_keys = {attr: val for attr, val in self.part_keys.items() if attr in self.datatypes}

    @classmethod
    def import_default(cls) -> LogProtocol:
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
        return cls.import_ini(LOG_PROTOCOL)

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

    def load_data(self, path: str, idxs: Optional[Iterable[int]]=None,
                  return_idxs=False) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Retrieve the main data array from the log file.

        Args:
            path : Path to the log file.
            idxs : Array of data indices to load. Loads info for all
                the frames by default.

        Returns:
            Dictionary with data fields and their names retrieved
            from the log file.
        """
        if idxs is not None:
            idxs = np.asarray(idxs)
            idxs.sort()

        line_count = 0
        with open(path, 'r') as log_file:
            for line_idx, line in enumerate(log_file):
                if line.startswith('# '):
                    if 'WARNING' not in line:
                        keys_line = line.strip('# ')
                else:
                    data_line = line

                    if idxs is None:
                        skiprows = line_idx
                        max_rows = None
                        break

                    if idxs.size == 0:
                        skiprows = line_idx
                        max_rows = 0
                        break

                    if line_count == idxs[0]:
                        skiprows = line_idx
                    if line_count == idxs[-1]:
                        max_rows = line_idx - skiprows + 1
                        break

                    line_count += 1

        keys = keys_line.strip('\n').split(';')
        data_strings = data_line.strip('\n').split(';')

        dtypes = {'names': [], 'formats': []}
        converters = {}
        for idx, (key, val) in enumerate(zip(keys, data_strings)):
            dtypes['names'].append(key)
            unit = self._get_unit(key)
            if 'float' in key:
                dtypes['formats'].append(np.dtype(float))
                converters[idx] = lambda item, unit=unit: unit * float(item)
            elif 'int' in key:
                if self._has_unit(key):
                    converters[idx] = lambda item, unit=unit: unit * float(item)
                    dtypes['formats'].append(np.dtype(float))
                else:
                    dtypes['formats'].append(np.dtype(int))
            elif 'Array' in key:
                dtypes['formats'].append(np.ndarray)
                func = lambda part, unit=unit: unit * float(part)
                conv = lambda item, func=func: np.asarray(list(map(func, item.strip(b' []').split(b','))))
                converters[idx] = conv
            else:
                dtypes['formats'].append('<S' + str(len(val)))
                converters[idx] = lambda item: item.strip(b' []')

        txt_dict = {}
        txt_tuple = np.loadtxt(path, delimiter=';', converters=converters,
                               dtype=dtypes, unpack=True, skiprows=skiprows,
                               max_rows=max_rows)

        if idxs is None:
            txt_dict.update(zip(keys, txt_tuple))
            idxs = np.arange(txt_tuple[0].size)
        elif idxs.size == 0:
            txt_dict.update(zip(keys, txt_tuple))
        else:
            txt_dict.update({key: np.atleast_1d(data)[idxs - np.min(idxs)]
                             for key, data in zip(keys, txt_tuple)})

        if return_idxs:
            return txt_dict, idxs
        return txt_dict

def converter_petra(dir_path: str, scan_num: int, idxs: Optional[Iterable[int]]=None,
                    **attributes: Any) -> CrystData:
    # log_prt = LogProtocol.import_default()

    h5_dir = os.path.join(dir_path, f'scan_frames/Scan_{scan_num:d}')
    # log_path = os.path.join(dir_path, f'server_log/Scan_logs/Scan_{scan_num:d}.log')
    h5_files = sorted([os.path.join(h5_dir, path) for path in os.listdir(h5_dir)
                       if path.endswith(('LambdaFar.nxs', '.h5'))])

    input_file = CXIStore(h5_files, mode='r')

    # log_attrs = log_prt.load_attributes(log_path)
    # log_data, idxs = log_prt.load_data(log_path, idxs=idxs, return_idxs=True)

    # n_frames = idxs.size
    # if n_frames:
    #     x_sample = log_attrs['Session logged attributes'].get('x_sample', 0.0)
    #     y_sample = log_attrs['Session logged attributes'].get('y_sample', 0.0)
    #     z_sample = log_attrs['Session logged attributes'].get('z_sample', 0.0)
    #     r_sample = log_attrs['Session logged attributes'].get('r_sample', 0.0)
    #     translations = np.tile([[x_sample, y_sample, z_sample]], (n_frames, 1))
    #     tilts = r_sample * np.ones(n_frames)
    #     for data_key, log_dset in log_data.items():
    #         for log_key in log_prt.log_keys['x_sample']:
    #             if log_key in data_key:
    #                 translations[:log_dset.size, 0] = log_dset
    #         for log_key in log_prt.log_keys['y_sample']:
    #             if log_key in data_key:
    #                 translations[:log_dset.size, 1] = log_dset
    #         for log_key in log_prt.log_keys['z_sample']:
    #             if log_key in data_key:
    #                 translations[:log_dset.size, 2] = log_dset
    #         for log_key in log_prt.log_keys['r_sample']:
    #             if log_key in data_key:
    #                 tilts[:log_dset.size] = log_dset

    #     return CrystData(input_file=input_file, translations=translations,
    #                      tilts=tilts, **attributes)

    return CrystData(input_file=input_file, **attributes)
