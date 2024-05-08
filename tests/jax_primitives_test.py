from typing import Callable, Tuple
import numpy as np
import pytest
from jax import config
from jax.test_util import check_grads
from cbclib.jax import (euler_angles, euler_matrix, tilt_angles, tilt_matrix, det_to_k,
                        k_to_det, k_to_smp, matmul)

class TestJaxPrimitives():
    Generator = Callable[[], np.ndarray]

    @pytest.fixture(params=[(5,), (5, 5)])
    def n_samples(self, request: pytest.FixtureRequest) -> Tuple[int, ...]:
        return request.param

    @pytest.fixture(params=[1, 4])
    def num_threads(self, request: pytest.FixtureRequest) -> int:
        return request.param

    @pytest.fixture
    def src(self, rng: np.random.Generator) -> Tuple[int, ...]:
        return tuple(rng.random(3))

    @pytest.fixture
    def generate_mats(self, rng: np.random.Generator, n_samples: Tuple[int, ...]) -> Generator:
        return lambda : rng.random(n_samples + (3, 3))

    @pytest.fixture
    def generate_vecs(self, rng: np.random.Generator, n_samples: Tuple[int, ...]) -> Generator:
        return lambda : rng.random(n_samples + (3,))

    @pytest.fixture
    def generate_coords(self, rng: np.random.Generator, n_samples: Tuple[int, ...]) -> Generator:
        return lambda : rng.random(n_samples)

    def check_gradient(self, f, args, num_threads: int):
        def wrapper(*args):
            return f(*args, num_threads=num_threads)
        check_grads(wrapper, args, order=1, modes='rev', eps=1e-5)

    def test_euler_angles(self, generate_mats: Generator, num_threads: int):
        self.check_gradient(euler_angles, (generate_mats(),), num_threads)

    def test_euler_matrix(self, generate_vecs: Generator, num_threads: int):
        self.check_gradient(euler_matrix, (generate_vecs(),), num_threads)

    def test_tilt_angles(self, generate_mats: Generator, num_threads: int):
        self.check_gradient(tilt_angles, (generate_mats(),), num_threads)

    def test_tilt_matrix(self, generate_vecs: Generator, num_threads: int):
        self.check_gradient(tilt_matrix, (generate_vecs(),), num_threads)

    def test_det_to_k(self, generate_coords: Generator, generate_vecs: Generator, num_threads: int):
        self.check_gradient(det_to_k, (generate_coords(), generate_coords(), generate_vecs()),
                            num_threads)

    def test_k_to_det(self, generate_vecs: Generator, num_threads: int):
        self.check_gradient(k_to_det, (generate_vecs(), generate_vecs()), num_threads)

    def test_k_to_smp(self, generate_coords: Generator, generate_vecs: Generator,
                      src: Tuple[int, int, int], num_threads: int):
        self.check_gradient(k_to_smp, (generate_vecs(), generate_coords(), src), num_threads)

    def test_matmul(self, generate_vecs: Generator, generate_mats: Generator, num_threads: int):
        self.check_gradient(matmul, (generate_vecs(), generate_mats()), num_threads=num_threads)
