from __future__ import annotations
from typing import List, Optional, Tuple, Union
import numpy as np
from .label import Points, Structure

Line = Tuple[float, float, float, float]

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
    points : Points

    def __init__(self, data: np.ndarray, mask: np.ndarray, radius: int, vmin: float):
        ...

    def filter(self, data: np.ndarray, mask: np.ndarray, structure: Structure, vmin: float,
               npts: int) -> Peaks:
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

    def mask(self, mask: np.ndarray) -> Peaks:
        """Discard all peaks that are not True in masking array.

        Args:
            mask : Boolean 2D array.

        Returns:
            A new masked set of peaks.
        """
        ...

def detect_peaks(data: np.ndarray, mask: np.ndarra, radius: int, vmin: float,
                 axes: Optional[Tuple[int, int]]=None, num_threads: int=1) -> List[Peaks]:
    ...

def filter_peaks(peaks: Union[Peaks, List[Peaks]], data: np.ndarray, mask: np.ndarray,
                 structure: Structure, vmin: float, npts: int, axes: Optional[Tuple[int, int]]=None,
                 num_threads: int=1) -> Union[Peaks, List[Peaks]]:
    ...

def detect_streaks(peaks: Union[Peaks, List[Peaks]], data: np.ndarray, mask: np.ndarray,
                   structure: Structure, xtol: float, vmin: float, log_eps: float=0.0, max_iter: int=100,
                   lookahead: int=1, min_size: int=5, axes: Optional[Tuple[int, int]]=None,
                   num_threads: int=1) -> Union[List[Line], List[List[Line]]]:
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
