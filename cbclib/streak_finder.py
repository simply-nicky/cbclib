from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Union
import numpy as np
from .data_container import DataContainer
from .cbc_setup import Streaks, ScanStreaks
from .cxi_protocol import Indices
from .src import detect_peaks, detect_streaks, filter_peaks, Structure, Peaks

Line = Tuple[float, float, float, float]

@dataclass
class CBSDetector(DataContainer):
    data : np.ndarray
    mask: np.ndarray
    structure : Structure
    num_threads : int

    def __post_init__(self):
        if self.data.ndim < 2:
            raise ValueError(f"Invalid number of dimensions: {self.data.ndim} != 2")
        if self.data.shape[-2:] != self.mask.shape:
            raise ValueError("data and mask have incompatible shapes: "\
                             f"{self.data.shape} != {self.mask.shape}")

    def __getitem__(self, idxs: Indices) -> CBSDetector:
        return self.replace(data=self.data[idxs], mask=self.mask[idxs])

    @property
    def shape(self) -> Tuple[int, int]:
        return self.data.shape

    def find_peaks(self, vmin: float, npts: int, connectivity: Structure=Structure(1, 1)) -> List[Peaks]:
        """Find peaks in a pattern. Returns a sparse set of peaks which values are above a threshold
        ``vmin`` that have a supporing set of a size larger than ``npts``. The minimal distance
        between peaks is ``2 * structure.radius``.

        Args:
            vmin : Peak threshold. All peaks with values lower than ``vmin`` are discarded.
            npts : Support size threshold. The support structure is a connected set of pixels which
                value is above the threshold ``vmin``. A peak is discarded is the size of support
                set is lower than ``npts``.
            connectivity : Connectivity structure used in finding a supporting set.

        Returns:
            Set of detected peaks.
        """
        peaks = detect_peaks(self.data, self.mask, 2 * self.structure.radius, vmin,
                             num_threads=self.num_threads)
        return filter_peaks(peaks, self.data, self.mask, connectivity, vmin, npts,
                            num_threads=self.num_threads)

    def detect(self, peaks: Union[Peaks, List[Peaks]], xtol: float, vmin: float, log_eps: float=0.0,
               max_iter: int=100, lookahead: int=3, min_size: int=5) -> Union[Streaks, ScanStreaks]:
        """Streak finding algorithm. Starting from the set of seed peaks, the lines are iteratively
        extended with a connectivity structure.

        Args:
            peaks : A set of peaks used as seed locations for the streak growing algorithm.
            xtol : Distance threshold. A new linelet is added to a streak if it's distance to the
                streak is no more than ``xtol``.
            vmin : Value threshold. A new linelet is added to a streak if it's value at the center
                of mass is above ``vmin``.
            log_eps : Detection threshold. A streak is added to the final list if it's number of
                false alarms (NFA) is above ``log_eps``.
            max_iter : Maximum number of iterations of the streak growing stage.
            lookahead : Number of linelets considered at the ends of a streak to be added to the
                streak.
            min_size : Minimum number of linelets required in a detected streak.
            
        Returns:
            A list of detected streaks.
        """
        if isinstance(peaks, list) and len(peaks) == 1:
            peaks = peaks[0]

        if isinstance(peaks, Peaks):
            streaks = np.array(detect_streaks(peaks, self.data, self.mask, self.structure, xtol,
                                              vmin, log_eps, max_iter, lookahead, min_size))
            return Streaks(*streaks.T)

        streaks = detect_streaks(peaks, self.data, self.mask, self.structure, xtol, vmin, log_eps,
                                 max_iter, lookahead, min_size, num_threads=self.num_threads)

        return ScanStreaks({idx: Streaks(*np.array(val).T) for idx, val in enumerate(streaks)})
