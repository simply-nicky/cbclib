from typing import List, Tuple, Union
import numpy as np

def euler_angles(rot_mats: np.ndarray) -> np.ndarray:
    ...

def euler_matrix(angles: np.ndarray) -> np.ndarray:
    ...

def tilt_matrix(tilts: np.ndarray, axis: Union[List[int], Tuple[int, ...], np.ndarray]) -> np.ndarray:
    ...

def find_rotations(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ...

def cartesian_to_spherical(vecs: np.ndarray) -> np.ndarray:
    ...

def spherical_to_cartesian(vecs: np.ndarray) -> np.ndarray:
    ...

def gaussian_grid(x_arr: np.ndarray, y_arr: np.ndarray, z_arr: np.ndarray, basis: np.ndarray,
                  Sigma: float, sigma: float, threads: int=1) -> np.ndarray:
    ...

def gaussian_grid_grad(x_arr: np.ndarray, y_arr: np.ndarray, z_arr: np.ndarray, basis: np.ndarray,
                       hkl: np.ndarray, Sigma: float, sigma: float, threads: int=1) -> np.ndarray:
    ...

def calc_source_lines(basis: np.ndarray, hkl: np.ndarray, kin_min: np.ndarray, kin_max: np.ndarray,
                      threads: int=1) -> np.ndarray:
    ...

def cross_entropy(x: np.ndarray, p: np.ndarray, q: np.ndarray, q_abs: float, epsilon: float) -> float:
    ...
