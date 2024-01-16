from __future__ import annotations
from typing import List, Tuple
import numpy as np

Line = Tuple[float, float, float, float]

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

class Peaks:
    """Peak finding algorithm. Finds sparse peaks in a two-dimensional image.

    Args:
        data : A rasterised 2D image.
        radius : The minimal distance between peaks. At maximum one peak belongs
            to a single square in a radius x radius 2d grid.
        vmin : Peak is discarded if it's value is lower than ``vmin``.

    Attributes:
        size : Number of found peaks.
        x : x coordinates of peak locations.
        y : y coordinates of peak locations.
    """
    points : List[Tuple[int, int]]
    size : int
    x : List[int]
    y : List[int]

    def __init__(self, data: np.ndarray, radius: int, vmin: float):
        ...

    def filter(self, data: np.ndarray, structure: Structure, vmin: float, npts: int) -> Peaks:
        """Discard all the peaks the support structure of which is too small. The support
        structure is a connected set of pixels which value is above the threshold ``vmin``.
        A peak is discarded is the size of support set is lower than ``npts``.

        Args:
            data : A rasterised 2D image.
            structure : Connectivity structure.
            vmin : Threshold value.
            npts : Minimal size of support structure.

        Returns:
            A new filtered set of peaks.
        """
        ...

def detect_streaks(peaks: Peaks, data: np.ndarray, mask: np.ndarray, structure: Structure,
                   xtol: float, vmin: float, log_eps: float=np.log(1e-1), max_iter: int=100,
                   lookahead: int=1, min_size: int=5) -> List[Line]:
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
        log_eps : Detection threshold. A streak is added to the final list if it's number of false
            alarms (NFA) is above ``log_eps``.
        max_iter : Maximum number of iterations of the streak growing stage.
        lookahead : Number of linelets considered at the ends of a streak to be added to the streak.
        min_size : Minimum number of linelets required in a detected streak.
        
    Returns:
        A list of detected streaks.
    """
    ...
