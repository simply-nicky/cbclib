import numpy as np
import pytest
from cbclib.src import median, median_filter
from cbclib.annotations import NDBoolArray, NDIntArray, NDRealArray, Shape

class TestImageProcessing():
    @pytest.fixture(params=[(9, 19, 29), (50, 30)])
    def shape(self, request: pytest.FixtureRequest) -> Shape:
        return request.param

    @pytest.fixture()
    def input(self, rng: np.random.Generator, shape: Shape) -> NDRealArray:
        return rng.random(shape)

    @pytest.fixture()
    def mask(self, rng: np.random.Generator, shape: Shape) -> NDBoolArray:
        return np.asarray(rng.integers(0, 1, size=shape), dtype=bool)

    def test_median(self, input: NDRealArray, mask: NDBoolArray):
        axes = list(range(input.ndim))
        out = median(input, axis=axes)
        out2 = np.median(input, axis=axes)

        assert np.all(out == out2)

        out = median(input, mask, axis=axes)
        out2 = median(input.ravel(), mask.ravel())

        assert np.all(out == out2)
