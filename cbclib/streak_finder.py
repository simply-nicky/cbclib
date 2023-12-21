from dataclasses import dataclass
import numpy as np
from .data_container import DataContainer
from .src import Structure

@dataclass
class Pattern(DataContainer):
    data : np.ndarray
    structure : Structure
