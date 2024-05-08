from __future__ import annotations
from typing import Iterator, List, Optional, Tuple, Union
from ..annotations import CPPIntSequence, Line, NDRealArray, NDBoolArray

class PointsSet:
    x : List[int]
    y : List[int]
    size : int

    def __init__(self, x: CPPIntSequence, y: CPPIntSequence):
        ...

class Structure:
    """Pixel connectivity structure class. Defines a two-dimensional connectivity kernel.
    Used in peaks and streaks detection algorithms.

    Args:
        radius : Radius of connectivity kernel. The size of the kernel is (2 * radius + 1,
            2 * radius + 1).
        rank : Rank determines which elements belong to the connectivity kernel, i.e. are
            considered as neighbors of the central element. Elements up to a squared distance
            of raml from the center are considered neighbors. Rank may range from 1 (no adjacent
            elements are neighbours) to radius (all elements in (2 * radius + 1, 2 * radius + 1)
            square are neighbours).

    Attributes:
        size : Number of elements in the connectivity kernel.
        x : x indices of the connectivity kernel.
        y : y indices of the connectivity kernel.
    """
    radius : int
    rank : int
    size : int
    x : List[int]
    y : List[int]

    def __init__(self, radius: int, rank: int):
        ...

class Regions:
    shape : List[int]

    def __init__(self, shape: CPPIntSequence, regions: List[PointsSet]=[]):
        ...

    def __delitem__(self, idxs: Union[int, slice]):
        ...

    def __getitem__(self, idxs: Union[int, slice]) -> Union[PointsSet, Regions]:
        ...

    def __iter__(self) -> Iterator[PointsSet]:
        ...

    def __len__(self) -> int:
        ...

    def __setitem__(self, idxs: Union[int, slice], value: Union[PointsSet, Regions]):
        ...

    def append(self, value: PointsSet):
        ...

    def filter(self, structure: Structure, npts: int) -> Regions:
        ...

    def mask(self) -> NDBoolArray:
        ...

    def center_of_mass(self, data: NDRealArray) -> List[List[float]]:
        ...

    def gauss_fit(self, data: NDRealArray) -> List[List[float]]:
        ...

    def ellipse_fit(self, data: NDRealArray) -> List[List[float]]:
        ...

    def line_fit(self, data: NDRealArray) -> List[Line]:
        ...

    def moments(self, data: NDRealArray) -> List[List[float]]:
        ...

def label(mask: NDBoolArray, structure: Structure, npts: int=1,
          axes: Optional[Tuple[int, int]]=None, num_threads: int=1) -> List[Regions]:
    ...
