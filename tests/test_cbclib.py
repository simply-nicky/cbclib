import os
import shutil
from typing import Iterable
import hdf5plugin
import pytest
from datetime import datetime
import cbclib as cbc

@pytest.fixture(scope='session')
def temp_dir():
    now = datetime.now()
    path = now.strftime("temp_%m_%d_%H%M%S")
    os.mkdir(path)
    yield path
    shutil.rmtree(path)

@pytest.mark.maxwell
def test_maxwell(dir_path: str, scan_num: int, roi: Iterable[int], temp_dir: str) -> None:
    assert os.path.isdir(temp_dir)
    data = cbc.converter_petra(dir_path, scan_num, os.path.join(temp_dir, 'test.h5'),
                               transform=cbc.Crop(roi=roi))
    assert os.path.isfile(os.path.join(temp_dir, 'test.h5'))
    assert 'data' in data.files
    data = data.load('data')

    data = data.update_mask(method='range-bad', vmax=10000000)
    assert data.get('mask') is not None
    data = data.update_whitefield(method='mean')
    assert data.get('whitefield') is not None

    data.save(attributes=['mask', 'whitefield', 'data', 'translations', 'tilts'])