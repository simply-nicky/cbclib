"""Log protocol (:class:`cbclib.LogProtocol`) together with log container (:class:`cbclib.LogContainer`)
provide an interface to retrieve the data from the log files, which contain the readouts from the
motors and other instrument during the experiment.

Examples:
    Generate a default built-in log protocol:

    >>> import cbclib as cbc
    >>> cbc.LogProtocol()
    LogProtocol(datatypes={'exposure': 'float', 'n_points': 'int', 'n_steps': 'int', 'scan_type':
    'str', 'step_size': 'float', 'x_sample': 'float', 'y_sample': 'float', 'z_sample': 'float',
    'r_sample': 'float'}, log_keys={'exposure': ['Exposure'], 'n_points': ['Points count'],
    'n_steps': ['Steps count'], 'scan_type': ['Device'], 'step_size': ['Step size'], 'x_sample':
    ['X-SAM', 'SAM-X', 'SCAN-X'], 'y_sample': ['Y-SAM', 'SAM-Y', 'SCAN-Y'], 'z_sample': ['Z-SAM',
    'SAM-Z', 'SCAN-Z'], 'r_sample': ['R-SAM', 'SAM-R', 'SCAN-R']}, part_keys={'exposure':
    'Type: Method', 'n_points': 'Type: Scan', 'n_steps': 'Type: Scan', 'scan_type': 'Type: Scan',
    'step_size': 'Type: Scan', 'x_sample': 'Session logged attributes', 'y_sample':
    'Session logged attributes', 'z_sample': 'Session logged attributes', 'r_sample':
    'Session logged attributes'})

    Generate a default log data container:

    >>> cbc.LogContainer()
    LogContainer(protocol=LogProtocol(datatypes={'exposure': 'float', 'n_points': 'int', 'n_steps':
    'int', 'scan_type': 'str', 'step_size': 'float', 'x_sample': 'float', 'y_sample': 'float',
    'z_sample': 'float', 'r_sample': 'float'}, log_keys={'exposure': ['Exposure'], 'n_points':
    ['Points count'], 'n_steps': ['Steps count'], 'scan_type': ['Device'], 'step_size': ['Step size'],
    'x_sample': ['X-SAM', 'SAM-X', 'SCAN-X'], 'y_sample': ['Y-SAM', 'SAM-Y', 'SCAN-Y'], 'z_sample':
    ['Z-SAM', 'SAM-Z', 'SCAN-Z'], 'r_sample': ['R-SAM', 'SAM-R', 'SCAN-R']}, part_keys={'exposure':
    'Type: Method', 'n_points': 'Type: Scan', 'n_steps': 'Type: Scan', 'scan_type': 'Type: Scan',
    'step_size': 'Type: Scan', 'x_sample': 'Session logged attributes', 'y_sample':
    'Session logged attributes', 'z_sample': 'Session logged attributes', 'r_sample':
    'Session logged attributes'}), log_attr={}, log_data={}, idxs=None, translations=None)
"""
from __future__ import annotations
from dataclasses import dataclass, field
import os
import re
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Tuple, TypeVar
import numpy as np
from .data_container import DataContainer, INIContainer
from .cbc_setup import Sample, ScanSamples, ScanSetup

LOG_PROTOCOL = os.path.join(os.path.dirname(__file__), 'config/log_protocol.ini')
L = TypeVar('L', bound='LogContainer')

@dataclass
class LogProtocol(INIContainer):
    """Log file protocol class. Contains log file keys to retrieve
    and the data types of the corresponding values.

    Args:
        datatypes : Dictionary with attributes' datatypes. 'float', 'int', 'bool', or 'str' are
            allowed.
        log_keys : Dictionary with attributes' log file keys.
        part_keys : Dictionary with the part names inside the log file where the attributes are
            stored.
    """
    __ini_fields__ = {'datatypes': 'datatypes', 'log_keys': 'log_keys', 'part_keys': 'part_keys'}

    datatypes : Dict[str, str]
    log_keys : Dict[str, List[str]]
    part_keys : Dict[str, str]

    known_types: ClassVar[Dict[str, Any]] = {'int': int, 'bool': bool, 'float': float, 'str': str}
    unit_dict: ClassVar[Dict[str, float]] = {'mm': 1e-3, 'mdeg': 1.7453292519943296e-05,
                                             'µm,um': 1e-6, 'udeg,µdeg': 1.7453292519943296e-08,
                                             'nm': 1e-9, 'ndeg': 1.7453292519943296e-11,
                                             'pm': 1e-12, 'pdeg': 1.7453292519943296e-14,
                                             'percent': 1e-2}

    def __post_init__(self):
        self.log_keys = {attr: self.str_to_list(val)
                         for attr, val in self.log_keys.items() if attr in self.datatypes}
        self.part_keys = {attr: val for attr, val in self.part_keys.items()
                          if attr in self.datatypes}

    @classmethod
    def import_default(cls) -> LogProtocol:
        """Return the default :class:`LogProtocol` object.

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

    def load_attributes(self, path: str) -> Dict[str, Dict[str, Any]]:
        """Return attributes' values from a log file at the given `path`.

        Args:
            path : Path to the log file.

        Returns:
            Dictionary with the attributes retrieved from the log file.
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
            idxs : Array of data indices to load. Loads info for all the frames by default.
            return_idxs : Return an array of indices of the scan steps read from the log file
                if True.

        Returns:
            A tuple of two elements:

            * Dictionary with data fields and their names retrieved from the log file.
            * An array of indices of the scan steps read from the log file.
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

@dataclass
class LogContainer(DataContainer):
    """Log data container class. Takes a log protocol :class:`cbclib.LogProtocol` and provides
    an interface to read the log files and generate a an array of sample translations and a set
    of scan samples :class:`cbclib.ScanSamples`.

    Args:
        protocol : A log protocol object
        log_attr : A dictionary of log attributes imported from a log file.
        log_data : A dictionary of log data imported from a log file.
        idxs : A set of indices of the scan steps imported from a log file.
        translations : An array of sample translations.
    """
    protocol        : LogProtocol = field(default_factory=LogProtocol.import_default)
    log_attr        : Dict[str, Dict[str, Any]] = field(default_factory=dict)
    log_data        : Dict[str, Any] = field(default_factory=dict)
    idxs            : Optional[np.ndarray] = None
    translations    : Optional[np.ndarray] = None

    _no_data_exc    : ClassVar[ValueError] = ValueError('No log data in the container')

    def __len__(self) -> int:
        return 0 if self.idxs is None else self.idxs.size

    def read_logs(self: L, log_path: str, idxs: Optional[Iterable[int]]=None) -> L:
        """Read a log file under the path `log_path`. Read out only the frame indices defined by
        ``idxs``. If ``idxs`` is None, read the whole log file.

        Args:
            log_path : Path to the log file.
            idxs : List of indices to read. Read the whole log file if None.

        Returns:
            A new log container with ``log_attr``, ``log_data``, and ``idxs`` updated.
        """
        log_attr = self.protocol.load_attributes(log_path)
        log_data, idxs = self.protocol.load_data(log_path, idxs=idxs, return_idxs=True)
        return LogContainerFull(**dict(self, log_attr=log_attr, log_data=log_data, idxs=idxs))

    def find_log_part_key(self, attr: str) -> Optional[str]:
        """Find a name of the log dictionary corresponding to an attribute name `attr`.

        Args:
            attr : A name of the attribute to find.

        Returns:
            A name of the log dictionary, corresponding to the given attribute name `attr`.
        """
        log_keys = self.protocol.log_keys.get(attr, [])
        for part in self.log_attr:
            for log_key in log_keys:
                if log_key in part:
                    return part
        return None

    def find_log_attribute(self, attr: str, part_key: Optional[str]=None) -> Optional[Any]:
        """Find a value in the log attributes corresponding to an attribute name `attr`.

        Args:
            attr : A name of the attribute to find.
            part_key : Search in the given part of the log dictionary if provided.

        Returns:
            Value of the log attribute. Returns None if nothing is found.
        """
        if part_key is None:
            part_key = self.protocol.part_keys.get(attr, '')
        part_dict = self.log_attr.get(part_key, {})
        value = part_dict.get(attr, None)
        return value

    def find_log_dataset(self, attr: str) -> Optional[np.ndarray]:
        """Find a dataset in the log data corresponding to an attribute name `attr`.

        Args:
            attr : A name of the attribute to find.

        Returns:
            Dataset for the given attribute. Returns None if nothing is found.
        """
        log_keys = self.protocol.log_keys.get(attr, [])
        for data_key, log_dset in self.log_data.items():
            for log_key in log_keys:
                if log_key in data_key:
                    return log_dset
        return None

    def simulate_translations(self: L) -> L:
        """Simulate sample translations based on the log attributes.

        Raises:
            ValueError : If ``log_attr`` is missing.

        Returns:
            A new log container with ``translations`` updated.
        """
        raise self._no_data_exc

    def read_translations(self: L) -> L:
        """Generate sample translations based on the log data.

        Raises:
            ValueError : If ``log_data`` is missing.

        Returns:
            A new log container with ``translations`` updated.
        """
        raise self._no_data_exc

    def generate_samples(self: L, dist: float, setup: ScanSetup) -> L:
        """Generate a :class:`cbclib.ScanSamples` object from the sample translations.

        Args:
            dist : Initial focus-to-sample distance in meters.
            setup : Experimental setup.

        Raises:
            ValueError : If ``translations`` is missing.

        Returns:
            A scan samples object.
        """
        raise self._no_data_exc

@dataclass
class LogContainerFull(LogContainer):
    protocol        : LogProtocol

    log_attr        : Dict[str, Dict[str, Any]]
    log_data        : Dict[str, Any]
    idxs            : Optional[np.ndarray] = None
    translations    : Optional[np.ndarray] = None

    def _is_log_translations(self) -> bool:
        return (self.find_log_attribute('x_sample') is not None and
                self.find_log_attribute('y_sample') is not None and
                self.find_log_attribute('z_sample') is not None and
                self.find_log_attribute('r_sample') is not None and
                (self.find_log_dataset('x_sample') is not None or
                 self.find_log_dataset('y_sample') is not None or
                 self.find_log_dataset('z_sample') is not None or
                 self.find_log_dataset('r_sample') is not None))

    def _is_sim_translations(self) -> bool:
        return (self.find_log_attribute('x_sample') is not None and
                self.find_log_attribute('y_sample') is not None and
                self.find_log_attribute('z_sample') is not None and
                self.find_log_attribute('r_sample') is not None and
                (self.find_log_part_key('x_sample') is not None or
                 self.find_log_part_key('y_sample') is not None or
                 self.find_log_part_key('z_sample') is not None or
                 self.find_log_part_key('r_sample') is not None))

    def simulate_translations(self) -> LogContainerFull:
        if not self._is_sim_translations():
            raise ValueError('The necessary data is not found')

        translations = np.tile((self.find_log_attribute('x_sample'),
                                self.find_log_attribute('y_sample'),
                                self.find_log_attribute('z_sample'),
                                self.find_log_attribute('r_sample')), (len(self), 1))
        translations = np.nan_to_num(translations)

        step_sizes, n_steps = [], []
        for scan_motor, unit_vec in zip(['x_sample', 'y_sample',
                                         'z_sample', 'r_sample'], np.eye(4, 4)):
            part_key = self.find_log_part_key(scan_motor)
            if part_key is not None:
                step_sizes.append(self.log_attr[part_key].get('step_size') * unit_vec)
                n_steps.append(self.log_attr[part_key].get('n_points'))

        steps = np.tensordot(np.stack(np.mgrid[[slice(0, n) for n in n_steps]], axis=0),
                             np.stack(step_sizes, axis=0), (0, 0)).reshape(-1, 4)
        return self.replace(translations=translations + steps)

    def read_translations(self) -> LogContainerFull:
        if not self._is_log_translations():
            raise ValueError('The necessary data is not found')

        translations = np.tile((self.find_log_attribute('x_sample'),
                                self.find_log_attribute('y_sample'),
                                self.find_log_attribute('z_sample'),
                                self.find_log_attribute('r_sample')), (len(self), 1))
        translations = np.nan_to_num(translations)

        for idx, scan_motor in enumerate(['x_sample', 'y_sample', 'z_sample', 'r_sample']):
            dset = self.find_log_dataset(scan_motor)
            if dset is not None:
                translations[:dset.size, idx] = dset
        return self.replace(translations=translations)

    def generate_samples(self, dist: float, setup: ScanSetup) -> ScanSamples:
        if self.translations is None:
            raise ValueError('No translations in the container')

        samples = {}
        for frame, translation in zip(self.idxs, self.translations):
            samples[frame] = Sample(setup.tilt_rotation(translation[3] - self.translations[0, 3]),
                                    translation[2] - self.translations[0, 2] + dist)
        return ScanSamples(samples)
