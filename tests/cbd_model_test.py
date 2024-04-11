from typing import Callable, Tuple, cast
import jax.numpy as jnp
import numpy as np
import pytest
import cbclib as cbc
from cbclib.annotations import JaxArray, JaxIntArray, NDIntArray

class TestData():
    a_vec           = [-0.00093604,-0.00893389,-0.00049815]
    b_vec           = [ 0.00688718,-0.00039195,-0.00573877]
    c_vec           = [ 0.01180108,-0.00199115, 0.01430421]
    foc_pos         = [ 0.14292289, 0.16409828,-0.39722229]
    roi             = (1100, 3260, 1040, 3108)
    pupil_roi       = [2211.13555081,2360.12483269,1952.07580504,2093.26351828]
    rot_axis        = [ 1.57123908,-1.56927065]
    rotations       = [[[ 9.9619448e-01,  3.2783555e-05, -8.7157689e-02],
                        [-4.4395445e-05,  1.0000000e+00, -1.3129010e-04],
                        [ 8.7157689e-02,  1.3465989e-04,  9.9619448e-01]],
                       [[ 9.8480743e-01,  5.3705335e-05, -1.7364986e-01],
                        [-1.0006334e-04,  9.9999994e-01, -2.5820805e-04],
                        [ 1.7364983e-01,  2.7166118e-04,  9.8480743e-01]],
                       [[ 9.6592510e-01,  6.2607127e-05, -2.5882185e-01],
                        [-1.6658218e-04,  9.9999988e-01, -3.7979320e-04],
                        [ 2.5882179e-01,  4.0996689e-04,  9.6592498e-01]]]
    smp_dist        = 0.006571637911728528
    dist            = -0.3906506481396024
    wavelength      = 7.09291721831675e-11
    x_pixel_size    = 7.5e-05
    y_pixel_size    = 7.5e-05

    @classmethod
    def basis(cls) -> cbc.Basis:
        return cbc.Basis(np.asarray(cls.a_vec), np.asarray(cls.b_vec), np.asarray(cls.c_vec))

    @classmethod
    def setup(cls) -> cbc.ScanSetup:
        return cbc.ScanSetup(np.asarray(cls.foc_pos), np.asarray(cls.pupil_roi),
                             np.asarray(cls.rot_axis), cls.smp_dist,
                             cls.wavelength, cls.x_pixel_size, cls.y_pixel_size)

    @classmethod
    def samples(cls) -> cbc.ScanSamples:
        return cbc.ScanSamples(np.arange(len(cls.rotations)), np.array(cls.rotations),
                               np.full(len(cls.rotations), cls.dist))

class TestCBDModel():
    Criterion = Callable[[cbc.Basis, cbc.ScanSamples, cbc.ScanSetup], float]
    JaxCriterion = Callable[[cbc.jax.Basis, cbc.jax.ScanSamples, cbc.jax.ScanSetup], JaxArray]

    @pytest.fixture(params=[1.0])
    def width(self, request: pytest.FixtureRequest) -> float:
        return request.param

    @pytest.fixture
    def crop(self) -> cbc.Crop:
        return cbc.Crop(TestData.roi)

    @pytest.fixture
    def model(self, crop: cbc.Crop) -> cbc.CBDModel:
        return cbc.CBDModel(TestData.basis(), TestData.samples(), TestData.setup(), crop)

    @pytest.fixture
    def hkl(self, model: cbc.CBDModel) -> NDIntArray:
        return model.basis.generate_hkl(0.3)

    @pytest.fixture
    def indices(self, model: cbc.CBDModel, hkl: NDIntArray) -> Tuple[NDIntArray, NDIntArray]:
        return model.filter_hkl(hkl)

    def criterion(self, hkl: NDIntArray, indices: Tuple[NDIntArray, NDIntArray], crop: cbc.Crop,
                  width: float) -> Criterion:
        hidxs, bidxs = indices

        def wrapper(basis: cbc.Basis, samples: cbc.ScanSamples, setup: cbc.ScanSetup) -> float:
            model = cbc.CBDModel(basis, samples, setup, crop)
            streaks = model.generate_streaks(hkl, hidxs, bidxs)
            return cast(float, np.mean(np.concatenate(streaks.to_lines(width))))

        return wrapper

    def criterion_jax(self, hkl: JaxIntArray, indices: Tuple[JaxIntArray, JaxIntArray],
                      crop: cbc.Crop, width: float) -> JaxCriterion:
        hidxs, bidxs = indices

        def wrapper(basis: cbc.jax.Basis, samples: cbc.jax.ScanSamples, setup: cbc.jax.ScanSetup):
            model = cbc.jax.CBDModel(basis, samples, setup, crop)
            streaks = model.generate_streaks(hkl, hidxs, bidxs)
            return jnp.mean(jnp.concatenate(streaks.to_lines(width)))

        return wrapper
