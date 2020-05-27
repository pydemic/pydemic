from typing import TYPE_CHECKING, Type, Tuple, Iterator

import sidekick as sk

from .params_info import ParamsInfo
from .. import utils

if TYPE_CHECKING:
    from ..models import Model


class Meta:
    """
    Meta information about model.

    Attributes:
        params:
            Information about parameters, subclass of :cls:`ParamsInfo`
    """

    cls: Type["Model"]
    params: "ParamsInfo"
    variables: Tuple[str]
    ndim: int

    @classmethod
    def from_arguments(cls, kind, bases, meta):
        """
        Create Meta instance for the given kind from the list of base classes and
        an inner Django-like Meta class declaration.
        """

        kwargs = meta_arguments(bases, meta)
        return Meta(kind, **kwargs)

    def __init__(self, cls, **kwargs):
        self.cls = cls
        self.explicit_kwargs = kwargs
        self.variables = tuple(p.name for p in sorted(iter_state_variables(cls)))
        self.ndim = len(self.variables)
        self.params = ParamsInfo(cls)
        self.data_aliases = data_aliases(cls, kwargs.pop("data_aliases", None))

        if kwargs:
            raise TypeError(f"invalid arguments: {set(kwargs)}")

    @sk.lazy
    def component_index(self):
        cls = self.cls
        if hasattr(cls, "DATA_COLUMNS"):
            items = zip(cls.DATA_COLUMNS, cls.DATA_COLUMNS)
        else:
            items = cls.DATA_ALIASES.items()

        idx_map = {}
        for i, (k, v) in enumerate(items):
            idx_map[k] = idx_map[v] = i
        return idx_map

    @sk.lazy
    def data_columns(self):
        cls = self.cls
        try:
            return tuple(getattr(cls, "DATA_COLUMNS"))
        except AttributeError:
            return tuple(cls.DATA_ALIASES.values())


def meta_arguments(bases, meta_declaration):
    """
    Extract keyword arguments to pass to a Meta declaration.
    """

    kwargs = {}
    for base in reversed(bases):
        try:
            meta: Meta = base._meta
        except AttributeError:
            continue
        else:
            kwargs.update(meta.explicit_kwargs)

    if meta_declaration is not None:
        ns = vars(meta_declaration)
        kwargs.update({k: v for k, v in ns.items() if not k.startswith("_")})
    return kwargs


def iter_state_variables(cls) -> Iterator[utils.state_property]:
    """
    Return a list of state variables for class.
    """
    for attr in dir(cls):
        value = getattr(cls, attr, None)
        if isinstance(value, utils.state_property):
            yield value


def data_aliases(cls, extra=None):
    """
    Return the data_aliases meta attribute for the given class.

    Extra is an optional dictionary with additional data alias definitions.
    """
    out = {}
    for base in reversed(cls.__bases__):
        print(base)
        try:
            out.update(base._meta.data_aliases)
        except AttributeError:
            continue

    out.update(extra or ())

    # Remove null aliases
    for k, v in list(out.items()):
        if v is None:
            del out[k]
    return out
