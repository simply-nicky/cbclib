from __future__ import annotations
import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import MISSING
from typing import (Any, ClassVar, Dict, Hashable, Iterable, List, Optional, Protocol, Tuple,
                    Type, TypeVar, Union, overload, runtime_checkable)
from typing_extensions import dataclass_transform
from jax.tree_util import register_pytree_with_keys
from ..annotations import PyTree
from ..data_container import DataclassInstance

T = TypeVar('T')

# NOTE: Actual return type is 'Field[T]', but we want to help type checkers
# to understand the magic that happens at runtime.
@overload
def field(*, static: bool=False, default: T, init: bool=...,
          repr: bool=..., hash: Optional[bool]=..., compare: bool=...,
          metadata: Optional[Mapping[str, Any]]=...) -> T:
    ...

@overload
def field(*, static: bool=False, default_factory: Callable[[], T], init: bool=...,
          repr: bool=..., hash: Optional[bool]=..., compare: bool=...,
          metadata: Optional[Mapping[str, Any]]=...) -> T:
    ...

@overload
def field(*, static: bool=False, init: bool=..., repr: bool=...,
          hash: Optional[bool]=..., compare: bool=...,
          metadata: Optional[Mapping[str, Any]]=...) -> Any:
    ...

def field(*, static: bool=False, default: Any=MISSING, default_factory: Any=MISSING,
          init: bool=True, repr: bool=True, hash: Optional[bool]=None,
          compare: bool=True, metadata: Optional[Mapping[str, Any]]=None) -> Any:
    """A field creator with a static indicator.

    The static flag indicates whether a field is a pytree or static.  Pytree fields are
    differentiated and traced.  Static fields are hashed and compared.
    """
    metadata_dict: Dict[str, Any] = {} if metadata is None else dict(metadata)
    metadata_dict['pytree_node'] = not static
    if default is MISSING:
        return dataclasses.field(default_factory=default_factory, init=init, repr=repr,
                                 hash=hash, compare=compare, metadata=metadata_dict)
    return dataclasses.field(default=default, init=init, repr=repr, hash=hash,
                             compare=compare, metadata=metadata_dict)

@runtime_checkable
class JaxDataclassInstance(DataclassInstance, Protocol):
    static_fields: ClassVar[list[str]]
    dynamic_fields: ClassVar[list[str]]

@overload
@dataclass_transform(field_specifiers=(field,))
def jax_dataclass(*, init: bool=True, repr: bool=True, eq: bool=True,
                  order: bool=False) -> Callable[[Type[Any]], Type[JaxDataclassInstance]]:
    ...

@overload
@dataclass_transform(field_specifiers=(field,))
def jax_dataclass(cls: Type[Any], /, *, init: bool=True, repr: bool=True,
                  eq: bool=True, order: bool=False) -> Type[JaxDataclassInstance]:
    ...

@dataclass_transform(field_specifiers=(field,))
def jax_dataclass(cls: Optional[Type[Any]]=None, /, *, init: bool=True, repr: bool=True,
                  eq: bool=True, order: bool=False
                  ) -> Union[Type[JaxDataclassInstance], Callable[[Type[Any]], Type[JaxDataclassInstance]]]:
    """A dataclass creator that creates a pytree.

    Returns the same class as was passed in, with dunder methods added based on the fields defined
    in the class.

    Examines PEP 526 annotations to determine fields.  Default values for fields are provided using
    assignment.

    To mark fields as static fields rather than JAX pytree fields, use the `field` function.
    In JAX, a static attribute is one that induces recompilation of a function when it changes, and
    consequently there is more flexibility about what can be done with such an attribute.

    For example::
    ```python
    from __future__ import annotations

    from typing import ClassVar

    import jax.numpy as jnp
    from tjax import Array, dataclass
    from tjax.dataclasses import field
    from jax import grad

    @dataclass
    class LearnedParameter:
        weight: Array
        constrain_positive: bool = field(static=True)
        minimum_positive_weight: ClassVar[Array] = 1e-6

        def trained(self,
                    self_bar: LearnedParameter,
                    learning_rate: float) -> LearnedParameter:
            weight_bar = self_bar.weight
            weight = self.weight - weight_bar * learning_rate
            if self.constrain_positive:
                weight = jnp.maximum(weight, self.minimum_positive_weight)
            return LearnedParameter(weight=weight,
                                    constrain_positive=self.constrain_positive)

    def loss(w: LearnedParameter) -> float:
        return jnp.square(w.weight - 3.3)

    w = LearnedParameter(2.0, True)
    w_bar = grad(loss)(w)
    new_w = w.trained(w_bar, 1e-4)
    ```

    Since this dataclass is a pytree, all of JAX's functions that accept pytrees work with it,
    including iteration, differentiation, and `jax.tree_util` functions.

    Another benefit is the display of dataclasses.  `print(new_w)` gives::
    ```
    LearnedParameter
        weight=Jax Array ()
                2.0003
        constrain_positive=True
    ```
    """
    if cls is None:
        def f(x: Type[Any], /) -> Type[JaxDataclassInstance]:
            return jax_dataclass(x, init=init, repr=repr, eq=eq, order=order)
        return f  # Type checking support partial is poor.
    non_none_cls = cls

    # Apply dataclass function to cls.
    data_clz = dataclasses.dataclass(init=init, repr=repr, eq=eq, order=order)(cls)

    # Partition fields into static, and dynamic; and assign these to the class.
    static_fields: List[str] = []
    dynamic_fields: List[str] = []
    for field_info in dataclasses.fields(data_clz):
        if field_info.metadata.get('pytree_node', True):
            dynamic_fields.append(field_info.name)
        else:
            static_fields.append(field_info.name)
    data_clz.static_fields = static_fields
    data_clz.dynamic_fields = dynamic_fields

    # Register the class as a Jax PyTree.
    def flatten_with_keys(x: Any, /) -> Tuple[Iterable[tuple[str, PyTree]], Hashable]:
        hashed = tuple(getattr(x, name) for name in static_fields)
        trees = tuple((name, getattr(x, name)) for name in dynamic_fields)
        return trees, hashed

    def unflatten(hashed: Hashable, trees: Iterable[PyTree], /) -> Any:
        if not isinstance(hashed, tuple):
            raise TypeError
        hashed_args = dict(zip(static_fields, hashed))
        tree_args = dict(zip(dynamic_fields, trees))
        return non_none_cls(**hashed_args, **tree_args)

    def flatten(x: Any, /) -> Tuple[Iterable[PyTree], Hashable]:
        hashed = tuple(getattr(x, name) for name in static_fields)
        trees = tuple(getattr(x, name) for name in dynamic_fields)
        return trees, hashed

    register_pytree_with_keys(data_clz, flatten_with_keys, unflatten, flatten)

    return data_clz
