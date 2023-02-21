import os
import shutil
from typing import List, Tuple
from datetime import datetime
import hdf5plugin
import pytest
import numpy as np
import cbclib as cbc

def generate_data(samples: cbc.ScanSamples, basis: cbc.Basis, setup: cbc.ScanSetup,
                  shape: Tuple[int, int, int], bgd_lvl: Tuple[float, float],
                  q_abs: float, stk_w: float, sgn_lvl: float, bad_n: int,
                  bad_lvl: Tuple[int, int]) -> np.ndarray:
    y, x = np.meshgrid(np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
    background = 1.0 - 2.0 * np.sqrt((x - shape[2] // 2)**2 + (y - shape[1] // 2)**2) / \
                 np.sqrt(shape[2]**2 + shape[1]**2)
    background = bgd_lvl[0] + background * (bgd_lvl[1] - bgd_lvl[0])
    hkl = basis.generate_hkl(q_abs)
    patterns = []
    for sample in samples.values():
        model = cbc.CBDModel(basis, sample, setup, shape=shape[1:])
        hkl = hkl[model.filter_hkl(hkl)]
        streaks = model.generate_streaks(hkl, stk_w)
        pattern = streaks.pattern_image(model.shape)
        patterns.append(pattern)
    data = sgn_lvl * np.stack(patterns) + background
    data = np.random.poisson(data)

    i = np.random.randint(0, shape[0] - 1, bad_n)
    j = np.random.randint(0, shape[1] - 1, bad_n)
    k = np.random.randint(0, shape[2] - 1, bad_n)
    data[i, j, k] = bad_lvl[0] + (bad_lvl[1] - bad_lvl[0]) * np.random.randint(bad_n)
    return data

@pytest.fixture(params=[{'lengths': [13e-3, 10e-3, 15e-3], 'angles': [0.3, -0.2, 0.15]},],
                scope='session')
def basis(request: pytest.FixtureRequest) -> cbc.Basis:
    mat = np.eye(3) * np.array(request.param['lengths'])
    rotation = cbc.Rotation.import_euler(request.param['angles'])
    return cbc.Basis.import_matrix(rotation(mat))

@pytest.fixture(params=[{'shape': (500, 500), 'det_dist': 0.1, 'smp_dist': 0.01,
                         'pupil_size': 25, 'pixel_size': 7.5e-5, 'energy': 1e4}],
                scope='session')
def setup(request: pytest.FixtureRequest) -> cbc.ScanSetup:
    wavelength = 12398.419297617678 / request.param['energy']
    foc_pos = np.array([0.5 * request.param['shape'][1] * request.param['pixel_size'],
                        0.5 * request.param['shape'][0] * request.param['pixel_size'],
                        request.param['det_dist']])
    pupil_roi = (0.5 * (request.param['shape'][1] - request.param['pupil_size']),
                 0.5 * (request.param['shape'][1] + request.param['pupil_size']),
                 0.5 * (request.param['shape'][0] - request.param['pupil_size']),
                 0.5 * (request.param['shape'][0] + request.param['pupil_size']))
    return cbc.ScanSetup(foc_pos=foc_pos, smp_dist=request.param['pixel_size'], pupil_roi=pupil_roi,
                         x_pixel_size=request.param['pixel_size'], wavelength=wavelength,
                         rot_axis=np.array([1.5707963267948966, 1.5707963267948966]),
                         y_pixel_size=request.param['pixel_size'])

@pytest.fixture(params=[{'n_frames': 50, 'foc_dist': 0.01, 'tilt': np.deg2rad(1.0)},],
                scope='session')
def samples(request: pytest.FixtureRequest, setup: cbc.ScanSetup) -> cbc.ScanSamples:
    samples = setup.tilt_samples(np.arange(request.param['n_frames']),
                                 request.param['tilt'] * np.arange(request.param['n_frames']))
    return samples

@pytest.fixture(params=[{'bgd_lvl': (5.0, 10.0), 'stk_w': 3.0, 'sgn_lvl': 10.0,
                         'bad_n': 300, 'bad_lvl': (200, 400), 'q_abs': 0.3},],
                scope='session')
def data(request: pytest.FixtureRequest, basis: cbc.Basis, samples: cbc.ScanSamples,
         setup: cbc.ScanSetup) -> np.ndarray:
    shape = (len(samples), int(2.0 * setup.foc_pos[1] / setup.y_pixel_size),
             int(2.0 * setup.foc_pos[0] / setup.x_pixel_size))
    return generate_data(**request.param, shape=shape, samples=samples, basis=basis, setup=setup)

@pytest.fixture(scope='session')
def temp_dir() -> str:
    now = datetime.now()
    path = now.strftime("temp_%m_%d_%H%M%S")
    os.mkdir(path)
    yield path
    shutil.rmtree(path)

@pytest.fixture(params=[{'load_paths': {'data': '/test/data/data', 'mask': '/test/data/mask'}},
                        {'load_paths': {}}],
                scope='session')
def cxi_protocol(request: pytest.FixtureRequest) -> cbc.CXIProtocol:
    default = dict(cbc.CXIProtocol.import_default())
    for attr, val in request.param.items():
        default[attr].update(val)
    return cbc.CXIProtocol(**default)

@pytest.mark.data_processing
def test_cxi_protocol(cxi_protocol: cbc.CXIProtocol, temp_dir: str):
    path = os.path.join(temp_dir, 'procotol.ini')
    cxi_protocol.to_ini(path)
    new_protocol = cbc.CXIProtocol.import_ini(path)
    assert new_protocol == cxi_protocol
    os.remove(path)

@pytest.fixture(scope='session')
def h5_path(data: np.ndarray, cxi_protocol: cbc.CXIProtocol, temp_dir: str) -> str:
    path = os.path.join(temp_dir, 'data.h5')
    with cbc.CXIStore(path, mode='w', protocol=cxi_protocol) as h5_file:
        h5_file.save_attribute('data', data)
    yield path
    os.remove(path)

def test_cxi_store(data: np.ndarray, cxi_protocol: cbc.CXIProtocol, temp_dir: str):
    path = os.path.join(temp_dir, 'test.h5')
    with cbc.CXIStore(path, mode='a', protocol=cxi_protocol) as h5_file:
        h5_file.save_attribute('data', data, mode='insert', idxs=np.arange(data.shape[0]))
        h5_file.save_attribute('data', data, mode='append')
    with cbc.CXIStore(path, mode='r', protocol=cxi_protocol) as h5_file:
        file_shape = (len(h5_file.indices()),) + h5_file.read_shape()
        new_data = h5_file.load_attribute('data', idxs=np.arange(data.shape[0], 2 * data.shape[0]))
    data_shape = (2 * data.shape[0], data.shape[1], data.shape[2])
    assert file_shape == data_shape
    assert np.all(new_data == data)
    os.remove(path)

@pytest.fixture(params=[{'roi': (20, 460, 45, 490)},
                        {'roi': (10, 440, 30, 495)}],
                scope='session')
def crop(request: pytest.FixtureRequest) -> cbc.Crop:
    return cbc.Crop(**request.param)

@pytest.fixture(scope='session')
def frames(data: np.ndarray) -> np.ndarray:
    frames = np.arange(data.shape[0])
    return np.concatenate((frames[:8:2], frames[8:]))

@pytest.fixture(scope='session')
def cryst_data(h5_path: str, cxi_protocol: cbc.CXIProtocol, crop: cbc.Crop, frames: np.ndarray) -> cbc.CrystData:
    data = cbc.CrystData(cbc.CXIStore(h5_path, 'r', cxi_protocol))
    with data.input_file:
        transform = cbc.ComposeTransforms((cbc.Mirror(0, data.input_file.read_shape()), crop))
    data = data.update_transform(transform)
    data = data.load(processes=4).mask_frames(frames)
    data = data.update_mask(method='range-bad', vmin=0, vmax=150)
    data = data.update_whitefield(method='median + mean', num_medians=5)
    return data

@pytest.mark.data_processing
def test_cryst_data(cryst_data: cbc.CrystData):
    assert np.sum(np.invert(cryst_data.mask).astype(int)) == np.sum((cryst_data.data > 150).astype(int))
    assert np.all(np.isclose(cryst_data.cor_data, (cryst_data.data - cryst_data.background) * cryst_data.mask))

@pytest.fixture(scope='session')
def lsd_detector(cryst_data: cbc.CrystData) -> cbc.LSDetector:
    lsd_det = cryst_data.lsd_detector()
    lsd_det = lsd_det.generate_patterns(vmin=0.0, vmax=10.0, size=(1, 3, 3))
    lsd_det = lsd_det.detect(cutoff=10.0, filter_threshold=1.0, group_threshold=0.9, dilation=3.5)
    lsd_det = lsd_det.update_patterns(dilations=(0.0, 3.5, 10.5))
    return lsd_det

@pytest.mark.data_processing
def test_lsd_detector(lsd_detector: cbc.LSDetector):
    assert lsd_detector.patterns.min() == 0.0 and lsd_detector.patterns.max() == 1.0

@pytest.fixture(scope='session')
def table_file(lsd_detector: cbc.LSDetector, temp_dir: str) -> str:
    path = os.path.join(temp_dir, 'table.h5')
    table = lsd_detector.export_table(concatenate=True)
    table.to_hdf(path, 'data')
    yield path
    os.remove(path)
