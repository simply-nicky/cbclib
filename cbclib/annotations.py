from typing import Any, List, Sequence, Tuple, Union
import numpy as np
import numpy.typing as npt
from jax import Array as JaxArray

Shape = Tuple[int, ...]

IntTuple = Tuple[int, ...]
FloatTuple = Tuple[float, ...]

JaxBoolArray = JaxArray
JaxIntArray = JaxArray
JaxRealArray = JaxArray
JaxComplexArray = JaxArray

NDArray = npt.NDArray[Any]
NDBoolArray = npt.NDArray[np.bool_]
NDIntArray = npt.NDArray[np.integer[Any]]
NDRealArray = npt.NDArray[np.floating[Any]]
NDComplexArray = npt.NDArray[Union[np.floating[Any], np.complexfloating[Any, Any]]]

Array = Union[JaxArray, NDArray]

Indices = Union[int, slice, NDIntArray, Sequence[int]]

IntSequence = Union[int, np.integer[Any], Sequence[int], NDIntArray]
ROIType = Union[List[int], Tuple[int, int, int, int], NDIntArray]
RealSequence = Union[float, np.floating[Any], Sequence[float], NDRealArray, JaxRealArray]

NDRealArrayLike = Union[NDRealArray, List[float], Tuple[float, ...]]
JaxRealArrayLike = Union[JaxArray, List[float], Tuple[float, ...]]

PyTree = Any
