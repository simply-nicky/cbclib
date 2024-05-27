from typing import Any, Dict, List, Literal, NamedTuple, Sequence, Set, Tuple, Union
import numpy as np
import numpy.typing as npt
from jax import Array as JaxArray

Scalar = Union[int, float, np.number]
Shape = Tuple[int, ...]

IntTuple = Tuple[int, ...]
FloatTuple = Tuple[float, ...]

JaxBoolArray = JaxArray
JaxIntArray = JaxArray
JaxRealArray = JaxArray
JaxComplexArray = JaxArray

NDArray = npt.NDArray[Any]
NDArrayLike = npt.ArrayLike
NDBoolArray = npt.NDArray[np.bool_]
NDIntArray = npt.NDArray[np.integer[Any]]
NDRealArray = npt.NDArray[np.floating[Any]]
NDComplexArray = npt.NDArray[Union[np.floating[Any], np.complexfloating[Any, Any]]]

Array = Union[JaxArray, NDArray]

Indices = Union[int, slice, NDIntArray, Sequence[int]]

IntSequence = Union[int, np.integer[Any], Sequence[int], NDIntArray]
ROIType = Union[List[int], Tuple[int, int, int, int], NDIntArray]
RealSequence = Union[float, np.floating[Any], Sequence[float], NDRealArray, JaxRealArray]
CPPIntSequence = Union[Sequence[int], NDIntArray]

NDRealArrayLike = Union[NDRealArray, List[float], Tuple[float, ...]]
JaxRealArrayLike = Union[JaxArray, List[float], Tuple[float, ...]]

PyTree = Any
Table = Dict[Tuple[int, int], float]

Norm = Literal['backward', 'forward', 'ortho']
Mode = Literal['constant', 'nearest', 'mirror', 'reflect', 'wrap']

Line = List[float]
Streak = Tuple[Set[Tuple[int, int, float]], Dict[float, List[float]],
               Dict[float, List[int]], Line]

class Pattern(NamedTuple):
    index   : NDIntArray
    frames  : NDIntArray
    y       : NDIntArray
    x       : NDIntArray

class PatternWithHKL(NamedTuple):
    index   : NDIntArray
    frames  : NDIntArray
    y       : NDIntArray
    x       : NDIntArray
    rp      : NDRealArray
    h       : NDIntArray
    k       : NDIntArray
    l       : NDIntArray

class PatternWithHKLID(NamedTuple):
    index   : NDIntArray
    frames  : NDIntArray
    y       : NDIntArray
    x       : NDIntArray
    rp      : NDRealArray
    h       : NDIntArray
    k       : NDIntArray
    l       : NDIntArray
    hkl_id  : NDIntArray
