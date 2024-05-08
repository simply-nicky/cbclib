from typing import Optional, Tuple
import numpy as np
import pytest
from cbclib import PatternStreakFinder, StreakFinderResult
from cbclib.src import draw_line_image, Peaks, PointsSet, Regions, Structure
from cbclib.annotations import NDBoolArray, NDIntArray, NDRealArray, Shape

class TestStreakFinder():
    atol = {np.dtype(np.float32): 1e-4, np.dtype(np.float64): 1e-5,
            np.dtype(np.complex64): 1e-4, np.dtype(np.complex128): 1e-5}
    rtol = {np.dtype(np.float32): 1e-3, np.dtype(np.float64): 1e-4,
            np.dtype(np.complex64): 1e-3, np.dtype(np.complex128): 1e-4}
    ATOL: float = 1e-8
    RTOL: float = 1e-5

    def check_close(self, a: np.ndarray, b: np.ndarray, rtol: Optional[float]=None,
                    atol: Optional[float]=None):
        if rtol is None:
            rtol = max(self.rtol.get(a.dtype, self.RTOL), self.rtol.get(b.dtype, self.RTOL))
        if atol is None:
            atol = max(self.atol.get(a.dtype, self.ATOL), self.atol.get(b.dtype, self.ATOL))
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)

    @pytest.fixture(params=[(50, 70)])
    def n_lines(self, request: pytest.FixtureRequest, rng: np.random.Generator) -> int:
        vmin, vmax = request.param
        return rng.integers(vmin, vmax)

    @pytest.fixture(params=[(80, 120, 2)])
    def shape(self, request: pytest.FixtureRequest, rng: np.random.Generator) -> Shape:
        vmin, vmax, size = request.param
        return tuple(rng.integers(vmin, vmax, size=size))

    @pytest.fixture(params=[30.0])
    def length(self, request: pytest.FixtureRequest) -> float:
        return request.param

    @pytest.fixture(params=[2.0])
    def width(self, request: pytest.FixtureRequest) -> float:
        return request.param

    @pytest.fixture
    def centers(self, rng: np.random.Generator, n_lines: int, shape: Shape) -> NDRealArray:
        return np.array([[shape[-1]], [shape[-2]]]) * rng.random((2, n_lines))

    @pytest.fixture
    def lines(self, rng: np.random.Generator, n_lines: int, centers: NDRealArray,
              length: float, width: float) -> NDRealArray:
        lengths = length * rng.random((n_lines,))
        thetas = 2 * np.pi * rng.random((n_lines,))
        x0, y0 = centers
        return np.stack((x0 - 0.5 * lengths * np.cos(thetas),
                         y0 - 0.5 * lengths * np.sin(thetas),
                         x0 + 0.5 * lengths * np.cos(thetas),
                         y0 + 0.5 * lengths * np.sin(thetas),
                         width * np.ones(n_lines)), axis=1)

    @pytest.fixture(params=[(0.15, 0.25)])
    def vmin(self, request: pytest.FixtureRequest, rng: np.random.Generator) -> float:
        vmin, vmax = request.param
        return vmin + (vmax - vmin) * rng.random()

    @pytest.fixture(params=[1.25])
    def xtol(self, request: pytest.FixtureRequest) -> float:
        return request.param

    @pytest.fixture(params=[-7])
    def log_eps(self, request: pytest.FixtureRequest) -> float:
        return request.param

    @pytest.fixture(params=[5])
    def npts(self, request: pytest.FixtureRequest) -> int:
        return request.param

    @pytest.fixture
    def image(self, lines: NDRealArray, shape: Shape, vmin: float,
              rng: np.random.Generator) -> NDRealArray:
        noise = 0.5 * vmin * rng.random(shape)
        return draw_line_image(lines, shape, kernel='biweight') + noise

    @pytest.fixture(params=[(0.04, 0.08)])
    def num_bad(self, request: pytest.FixtureRequest, shape: Shape,
                rng: np.random.Generator) -> int:
        vmin, vmax = request.param
        size = np.prod(shape)
        return int(rng.integers(int(vmin * size), int(vmax * size)))

    @pytest.fixture
    def mask(self, shape: Shape, num_bad: int, rng: np.random.Generator) -> NDBoolArray:
        mask = np.ones(shape, dtype=bool)
        indices = np.unravel_index(rng.integers(0, mask.size, num_bad), mask.shape)
        mask[indices] = False
        return mask

    @pytest.fixture(params=[(3, 2)])
    def structure(self, request: pytest.FixtureRequest) -> Structure:
        radius, rank = request.param
        return Structure(radius, rank)

    @pytest.fixture
    def finder(self, image: NDRealArray, mask: NDBoolArray, structure: Structure
               ) -> PatternStreakFinder:
        return PatternStreakFinder(image, mask, structure)

    @pytest.fixture
    def peaks(self, finder: PatternStreakFinder, vmin: float, npts: int) -> Peaks:
        return finder.detect_peaks(vmin, npts)

    @pytest.fixture
    def result(self, finder: PatternStreakFinder, peaks: Peaks, vmin: float, xtol: float,
               log_eps: float) -> StreakFinderResult:
        return finder.detect_streaks(peaks, xtol, vmin, log_eps)

    def get_pixels(self, x: int, y: int, finder: PatternStreakFinder) -> Tuple[NDIntArray, NDIntArray]:
        xs, ys = np.array(finder.structure.x) + x, np.array(finder.structure.y) + y
        mask = (xs >= 0) & (xs < finder.mask.shape[1]) & (ys >= 0) & (ys < finder.mask.shape[0])
        mask &= finder.mask[ys, xs]
        return xs[mask], ys[mask]

    def get_line(self, x: int, y: int, finder: PatternStreakFinder) -> NDRealArray:
        xvec, yvec = self.get_pixels(x, y, finder)
        region = Regions(finder.data.shape, [PointsSet(xvec, yvec)])
        return np.array(region.line_fit(finder.data)[0]).reshape(2, 2)

    def test_streak_points(self, result: StreakFinderResult, finder: PatternStreakFinder,
                           rng: np.random.Generator):
        index = rng.integers(0, len(result.streaks))
        pixels, pts, ctrs, _ = result.get_streak(index)
        linelets = np.concatenate([self.get_line(ctr[0], ctr[1], finder)
                                   for ctr in ctrs.values()], axis=0)
        streakpoints = np.stack(list(pts.values()))
        self.check_close(linelets[np.lexsort(linelets.T)], streakpoints[np.lexsort(streakpoints.T)])

        pts = np.unique(np.concatenate([self.get_pixels(ctr[0], ctr[1], finder)
                                        for ctr in ctrs.values()], axis=1).T, axis=0)
        pix_pts = np.array([[x, y] for x, y, _ in pixels])
        assert np.all(pts[np.lexsort(pts.T)] == pix_pts[np.lexsort(pix_pts.T)])

    def test_mask(self, result: StreakFinderResult, finder: PatternStreakFinder):
        assert np.all((result.mask != -1) == finder.mask)
