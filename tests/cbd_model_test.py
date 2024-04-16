import pytest
from typing import Any, Callable, Tuple
import jax.numpy as jnp
import jax
from jax.test_util import check_grads
import cbclib as cbc
from cbclib.annotations import JaxArray, JaxIntArray

jax.config.update("jax_enable_x64", True)

class Parameters():
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
    def basis(cls) -> cbc.jax.Basis:
        return cbc.jax.Basis(jnp.asarray(cls.a_vec), jnp.asarray(cls.b_vec), jnp.asarray(cls.c_vec))

    @classmethod
    def setup(cls) -> cbc.jax.ScanSetup:
        return cbc.jax.ScanSetup(jnp.asarray(cls.foc_pos), jnp.asarray(cls.pupil_roi),
                                 jnp.asarray(cls.rot_axis), cls.smp_dist,
                                 cls.wavelength, cls.x_pixel_size, cls.y_pixel_size)

    @classmethod
    def samples(cls) -> cbc.jax.ScanSamples:
        return cbc.jax.ScanSamples(jnp.arange(len(cls.rotations)), jnp.array(cls.rotations),
                                   jnp.full(len(cls.rotations), cls.dist))

class TestCBDModel():
    RTOL: float = 1e-3
    ATOL: float = 1e-5
    EPS: float = 1e-11
    Criterion = Callable[[cbc.jax.Basis, cbc.jax.ScanSamples, cbc.jax.ScanSetup], JaxArray]

    @pytest.fixture(params=[1.0])
    def width(self, request: pytest.FixtureRequest) -> float:
        return request.param

    @pytest.fixture
    def crop(self) -> cbc.Crop:
        return cbc.Crop(Parameters.roi)

    @pytest.fixture
    def model(self, crop: cbc.Crop) -> cbc.jax.CBDModel:
        return cbc.jax.CBDModel(Parameters.basis(), Parameters.samples(), Parameters.setup(), crop)

    @pytest.fixture
    def hkl(self, model: cbc.jax.CBDModel) -> JaxIntArray:
        return model.basis.generate_hkl(0.3)

    @pytest.fixture
    def indices(self, model: cbc.jax.CBDModel, hkl: JaxIntArray) -> Tuple[JaxIntArray, JaxIntArray]:
        return model.filter_hkl(hkl)

    @pytest.fixture
    def criterion(self, hkl: JaxIntArray, indices: Tuple[JaxIntArray, JaxIntArray],
                  crop: cbc.Crop, width: float) -> Criterion:
        hidxs, bidxs = indices

        def wrapper(basis: cbc.jax.Basis, samples: cbc.jax.ScanSamples, setup: cbc.jax.ScanSetup):
            model = cbc.jax.CBDModel(basis, samples, setup, crop)
            streaks = model.generate_streaks(hkl, hidxs, bidxs)
            return jnp.mean(jnp.concatenate(streaks.to_lines(width)))

        return wrapper

    def check_gradient(self, f: Callable, args: Tuple[Any, ...]):
        def wrapper(*args):
            return f(*args)
        check_grads(wrapper, args, order=1, modes='rev',
                    eps=self.EPS, rtol=self.RTOL, atol=self.ATOL)

    @pytest.mark.slow
    def test_model_gradients(self, criterion: Criterion, model: cbc.jax.CBDModel):
        self.check_gradient(criterion, (model.basis, model.samples, model.setup))

        self.check_gradient(lambda basis: criterion(basis, model.samples, model.setup),
                            (model.basis,))

        self.check_gradient(lambda samples: criterion(model.basis, samples, model.setup),
                            (model.samples,))

        self.check_gradient(lambda setup: criterion(model.basis, model.samples, setup),
                            (model.setup,))
