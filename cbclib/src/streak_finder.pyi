from typing import List, Tuple
import numpy as np

class Structure:
    radius : int
    rank : int
    idxs : List[Tuple[int, int]]

    def __init__(self, radius: int, rank: int):
        ...

class DetState:
    peaks : List[Tuple[int, int]]
    used : List[bool]

    def __init__(self, data: np.ndarray, radius: int, vmin: float):
        ...

    def filter(self, data: np.ndarray, structure: Structure, vmin: float, npts: int) -> DetState:
        ...
