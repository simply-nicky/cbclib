from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union
from ..annotations import CPPIntSequence, Line, NDBoolArray, NDIntArray, NDRealArray, Scalar, Streak
from .label import Structure

class Peaks:
    """Peak finding algorithm. Finds sparse peaks in a two-dimensional image.

    Args:
        data : A rasterised 2D image.
        mask : Mask of bad pixels. mask is False if the pixel is bad. Bad pixels are
            skipped in the peak finding algorithm.
        radius : The minimal distance between peaks. At maximum one peak belongs
            to a single square in a radius x radius 2d grid.
        vmin : Peak is discarded if it's value is lower than ``vmin``.

    Attributes:
        size : Number of found peaks.
        x : x coordinates of peak locations.
        y : y coordinates of peak locations.
    """
    x : List[int]
    y : List[int]
    size : int

    def __init__(self, x: CPPIntSequence, y: CPPIntSequence):
        ...

    def filter(self, data: NDRealArray, mask: NDBoolArray, structure: Structure,
               vmin: float, npts: int) -> Peaks:
        """Discard all the peaks the support structure of which is too small. The support
        structure is a connected set of pixels which value is above the threshold ``vmin``.
        A peak is discarded is the size of support set is lower than ``npts``.

        Args:
            data : A rasterised 2D image.
            mask : Mask of bad pixels. mask is False if the pixel is bad. Bad pixels are
                skipped in the peak finding algorithm.
            vmin : Threshold value.
            npts : Minimal size of support structure.

        Returns:
            A new filtered set of peaks.
        """
        ...

    def find_nearest(self, x: Scalar, y: Scalar) -> Tuple[List[int], float]:
        ...

    def find_range(self, x: Scalar, y: Scalar, range: float) -> List[Tuple[List[int], float]]:
        ...

    def mask(self, mask: NDBoolArray) -> Peaks:
        """Discard all peaks that are not True in masking array.

        Args:
            mask : Boolean 2D array.

        Returns:
            A new masked set of peaks.
        """
        ...

    def sort(self, data: NDRealArray):
        ...

def detect_peaks(data: NDRealArray, mask: NDBoolArray, radius: int, vmin: float,
                 axes: Optional[Tuple[int, int]]=None, num_threads: int=1) -> List[Peaks]:
    ...

def filter_peaks(peaks: List[Peaks], data: NDRealArray, mask: NDBoolArray,
                 structure: Structure, vmin: float, npts: int, axes: Optional[Tuple[int, int]]=None,
                 num_threads: int=1) -> List[Peaks]:
    ...

class StreakFinderResultDouble:
    mask : NDIntArray
    idxs : List[int]
    streaks : Dict[int, Line]

    def __init__(self, data: NDRealArray, mask: NDBoolArray):
        ...

    def get_streak(self, index: int) -> Streak:
        ...

    def probability(self, data: NDRealArray, vmin: float) -> float:
        ...

    def p_value(self, index: int, xtol: float, vmin: float, probability: float) -> float:
        ...

class StreakFinderResultFloat:
    mask : NDIntArray
    idxs : List[int]
    streaks : Dict[int, Line]

    def __init__(self, data: NDRealArray, mask: NDBoolArray):
        ...

    def get_streak(self, index: int) -> Streak:
        ...

    def probability(self, data: NDRealArray, vmin: float) -> float:
        ...

    def p_value(self, index: int, xtol: float, vmin: float, probability: float) -> float:
        ...

StreakFinderResult = Union[StreakFinderResultDouble, StreakFinderResultFloat]

class StreakFinder:
    structure   : Structure
    min_size    : int
    lookahead   : int
    nfa         : int

    def __init__(self, structure: Structure, min_size: int, lookahead: int=0, nfa: int=0):
        ...

    def detect_streaks(self, data: NDRealArray, mask: NDBoolArray, peaks: Peaks,
                       xtol: float, vmin: float) -> StreakFinderResult:
        ...

def detect_streaks(peaks: List[Peaks], data: NDRealArray, mask: NDBoolArray,
                   structure: Structure, xtol: float, vmin: float, min_size: int,
                   lookahead: int=0, nfa: int=0, axes: Optional[Tuple[int, int]]=None,
                   num_threads: int=1) -> List[List[Line]]:
    """Streak finding algorithm. Starting from the set of seed peaks, the lines are iteratively
    extended with a connectivity structure.

    Args:
        peaks : A set of peaks used as seed locations for the streak growing algorithm.
        data : A 2D rasterised image.
        mask : Mask of bad pixels. mask is False if the pixel is bad. Bad pixels are skipped in the
            streak detection algorithm.
        structure : A connectivity structure.
        xtol : Distance threshold. A new linelet is added to a streak if it's distance to the
            streak is no more than ``xtol``.
        vmin : Value threshold. A new linelet is added to a streak if it's value at the center of
            mass is above ``vmin``.
        log_eps : Detection threshold. A streak is added to the final list if it's p-value under
            null hypothesis is below ``np.exp(log_eps)``.
        lookahead : Number of linelets considered at the ends of a streak to be added to the streak.
        nfa : Number of false alarms, allowed number of unaligned points in a streak.

    Returns:
        A list of detected streaks.
    """
    ...
