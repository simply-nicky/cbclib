from typing import Callable, Dict, Optional
import numpy as np
import jax.numpy as jnp
from jax import config
import pytest
from jax.test_util import check_grads
from cbclib.src import draw_line_image, draw_line_table
from cbclib.jax import draw_line
from cbclib.annotations import NDIntArray, NDRealArray, Shape, Table

config.update("jax_enable_x64", True)

class TestDrawLine():
    ATOL: float = 1e-8
    RTOL: float = 1e-3

    def check_close(self, a: NDRealArray, b: NDRealArray, rtol: Optional[float]=None,
                    atol: Optional[float]=None):
        if rtol is None:
            rtol = self.RTOL
        if atol is None:
            atol = self.ATOL
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)

    @pytest.fixture(params=[(10, 50),])
    def n_lines(self, request: pytest.FixtureRequest, rng: np.random.Generator) -> int:
        vmin, vmax = request.param
        return rng.integers(vmin, vmax)

    @pytest.fixture(params=[(10, 50, 4),])
    def shape(self, request: pytest.FixtureRequest, rng: np.random.Generator) -> Shape:
        vmin, vmax, size = request.param
        return tuple(rng.integers(vmin, vmax, size=size))

    @pytest.fixture(params=[10.0])
    def length(self, request: pytest.FixtureRequest) -> float:
        return request.param

    @pytest.fixture(params=[2.0])
    def width(self, request: pytest.FixtureRequest) -> float:
        return request.param

    @pytest.fixture(params=["biweight", "gaussian", "parabolic", "rectangular", "triangular"])
    def kernel(self, request: pytest.FixtureRequest) -> str:
        return request.param

    @pytest.fixture(params=[1.0, 10.0])
    def max_val(self, request: pytest.FixtureRequest) -> float:
        return request.param

    @pytest.fixture
    def idxs(self, rng: np.random.Generator, n_lines: int, shape: Shape) -> NDIntArray:
        return rng.integers(0, np.prod(shape[:-2]) - 1, size=n_lines)

    @pytest.fixture
    def lines(self, rng: np.random.Generator, n_lines: int, shape: Shape,
              length: float, width: float) -> NDRealArray:
        lengths = length * rng.random((n_lines,))
        thetas = 2 * np.pi * rng.random((n_lines,))
        x0, y0 = np.array([[shape[-1]], [shape[-2]]]) * rng.random((2, n_lines))
        return np.stack((x0 - 0.5 * lengths * np.cos(thetas),
                         y0 - 0.5 * lengths * np.sin(thetas),
                         x0 + 0.5 * lengths * np.cos(thetas),
                         y0 + 0.5 * lengths * np.sin(thetas),
                         width * np.ones(n_lines)), axis=1)

    @pytest.fixture
    def image(self, lines: NDRealArray, idxs: NDIntArray, shape: Shape, max_val: float,
              kernel: str) -> NDRealArray:
        return draw_line_image(lines, shape, idxs, max_val=max_val, kernel=kernel)

    @pytest.fixture
    def table(self, lines: NDRealArray, idxs: NDIntArray, shape: Shape, max_val: float,
              kernel: str) -> Table:
        return draw_line_table(lines, shape, idxs, max_val=max_val, kernel=kernel)

    def test_empty_lines(self, shape: Shape):
        image = draw_line_image(np.zeros((0, 5)), shape[-2:])
        table = draw_line_table(np.zeros((0, 5)), shape[-2:])
        assert np.sum(image) == 0.0
        assert len(table) == 0

    @pytest.mark.xfail(raises=ValueError)
    def test_image_wrong_size_lines(self, lines: NDRealArray, idxs: NDIntArray, shape: Shape):
        image = draw_line_image(lines[::2], shape, idxs)

    @pytest.mark.xfail(raises=ValueError)
    def test_table_wrong_size_lines(self, lines: NDRealArray, idxs: NDIntArray, shape: Shape):
        table = draw_line_image(lines[::2], shape, idxs)

    def test_zero_width(self, lines: NDRealArray, idxs: NDIntArray, shape: Shape, kernel: str):
        zero_lines = np.concatenate((lines[..., :4], np.zeros(lines.shape[:-1] + (1,))), axis=-1)
        image = draw_line_image(zero_lines, shape, idxs, kernel=kernel)
        table = draw_line_table(zero_lines, shape, idxs, kernel=kernel)
        assert np.sum(image) == 0
        assert len(table) == 0

    def test_negative_width(self, lines: NDRealArray, idxs: NDIntArray, shape: Shape, kernel: str):
        neg_lines = np.concatenate((lines[..., :4], np.full(lines.shape[:-1] + (1,), -1)), axis=-1)
        image = draw_line_image(neg_lines, shape, idxs, kernel=kernel)
        table = draw_line_table(neg_lines, shape, idxs, kernel=kernel)
        assert np.sum(image) == 0
        assert len(table) == 0

    @pytest.mark.slow
    def test_max_val(self, image: NDRealArray, table: Table, n_lines: int, max_val: float):
        assert np.min(image) == 0
        assert np.all(np.max(image, axis=(-2, -1)) <= n_lines * max_val)
        assert np.min(list(table.values())) >= 0
        assert np.all(np.array(list(table.values())) <= max_val)

    def kernel_dict(self) -> Dict[str, Callable[[NDRealArray, NDRealArray], NDRealArray]]:
        def biweight(x, sigma):
            return 0.9375 * np.clip(1 - (x / sigma)**2, 0, np.infty)**2
        def gaussian(x, sigma):
            return np.where(np.abs(x) < sigma,
                            np.exp(-(3 * x / sigma)**2 / 2) / np.sqrt(2 * np.pi), 0)
        def parabolic(x, sigma):
            return 0.75 * np.clip(1 - (x / sigma)**2, 0, np.infty)
        def rectangular(x, sigma):
            return np.where(np.abs(x) < sigma, 1, 0)
        def triangular(x, sigma):
            return np.clip(1 - np.abs(x / sigma), 0, np.infty)

        return {'biweight': biweight, 'gaussian': gaussian, 'parabolic': parabolic,
                'rectangular': rectangular, 'triangular': triangular}

    @pytest.fixture
    def image_numpy(self, lines: NDRealArray, idxs: NDIntArray, shape: Shape,
                    max_val: float, kernel: str) -> np.ndarray:
        kernel_func = self.kernel_dict()[kernel]
        x, y = np.meshgrid(np.arange(shape[-1]), np.arange(shape[-2]))

        frames = []
        for fnum in range(np.prod(shape[:-2])):
            lns = lines[idxs == fnum]
            tau = np.array([lns[:, 2] - lns[:, 0], lns[:, 3] - lns[:, 1]])
            norm = np.array([lns[:, 3] - lns[:, 1], lns[:, 0] - lns[:, 2]])
            length = np.sqrt(np.sum(tau**2, axis=0))
            r1 = ((x[..., None] - lns[:, 0]) * norm[0] +
                  (y[..., None] - lns[:, 1]) * norm[1]) / length
            r2 = ((x[..., None] - lns[:, 0]) * tau[0] +
                  (y[..., None] - lns[:, 1]) * tau[1]) / length
            r2 = np.where(r2 < 0, r2, np.where(r2 > length, r2 - length, 0))
            frame = max_val * kernel_func(np.sqrt(r1 * r1 + r2 * r2), lns[:, 4])
            frames.append(np.sum(frame, axis=-1))
        return np.stack(frames).reshape(shape)

    def test_draw_line_image(self, image: NDRealArray, image_numpy: NDRealArray):
        self.check_close(image, image_numpy)

    def test_draw_line_table(self, table: Table, image_numpy: NDRealArray):
        image = np.zeros(image_numpy.shape)
        for (_, idx), val in table.items():
            image[np.unravel_index(idx, image_numpy.shape)] += val
        self.check_close(image, image_numpy)

    def test_draw_line_gradient(self, lines: NDRealArray, idxs: NDIntArray, shape: Shape,
                                max_val: float, kernel: str):
        if kernel == 'rectangular':
            pytest.xfail("rectangular kernel has no valid gradient")

        def wrapper(lines):
            return draw_line(jnp.zeros(shape), jnp.asarray(lines), jnp.asarray(idxs),
                             max_val=max_val, kernel=kernel)

        check_grads(wrapper, (lines,), order=1, modes='rev')
