from typing import (
    TYPE_CHECKING,
    Type,
    Tuple,
    Iterator,
    Dict,
    FrozenSet,
    Iterable,
    TypeVar,
    Sequence,
)

from .params_info import ParamsInfo
from .. import utils

if TYPE_CHECKING:
    from ..models import Model  # noqa: F401
    from ..models.metaclass import ModelMeta

T = TypeVar("T")


class Meta:
    """
    Meta information about model.

    Attributes:
        params:
            Information about parameters, subclass of :cls:`ParamsInfo`
    """

    cls: Type["Model"]
    model_name: str
    params: "ParamsInfo"
    variables: Tuple[str]
    data_aliases: Dict[str, str]
    plot_columns: FrozenSet[str]
    ndim: int

    @classmethod
    def from_arguments(cls, kind, meta):
        """
        Create Meta instance for the given kind from the list of base classes and
        an inner Django-like Meta class declaration.
        """

        if meta is not None:
            ns = vars(meta)
            kwargs = {k: v for k, v in ns.items() if not k.startswith("_")}
        else:
            kwargs = {}
        return Meta(kind, **kwargs)

    def __init__(self, cls, **kwargs):
        cls.meta = self
        self.cls = cls
        self.explicit_kwargs = kwargs.copy()
        self.variables = tuple(p.name for p in sorted(iter_state_variables(cls)))
        self.ndim = len(self.variables)
        self.params = ParamsInfo(cls)
        self.data_aliases = data_aliases(cls, kwargs.pop("data_aliases", None))

        # Keyword variables
        keywords = explicit_keywords(cls, self=True)
        self.model_name = keywords.get("model_name", "Model")

        # Plot columns
        new = kwargs.pop("plot_columns", (...,))
        bases = get_base_meta_attr(cls, "plot_columns", ())
        self.plot_columns = frozenset(merge_with_ellipsis(new, bases))

        # Check keyword args
        invalid = set(kwargs) - {"model_name", "plot_columns"}
        if invalid:
            raise TypeError(f"invalid arguments: {invalid}")

    def get_variable_index(self, name):
        """
        Return index from variable name or alias.
        """
        name = self.data_aliases.get(name, name)
        return self.variables.index(name)

    def __repr__(self):
        return f"<{self.cls.__name__}.meta object>"


def explicit_keywords(cls, self=False):
    """
    Extract explicit keyword arguments from meta object from all parent types.
    """

    kwargs = {}
    metas = list(iter_metas(cls, self=self))

    for meta in reversed(metas):
        kwargs.update(meta.explicit_kwargs)

    return kwargs


def iter_state_variables(cls) -> Iterator[utils.state_property]:
    """
    Return a list of state variables for class.
    """
    for attr in dir(cls):
        value = getattr(cls, attr, None)
        if isinstance(value, utils.state_property):
            yield value


def iter_metas(cls: "ModelMeta", self=False):
    """
    Iterate over _meta objects of parent classes.
    """
    if self:
        yield cls.meta

    for base in cls.mro()[1:]:
        if hasattr(base, "meta"):
            yield base.meta


def get_base_meta_attr(cls, prop, default):
    """
    Return attribute from the _meta attribute of a base class.
    """

    for meta in iter_metas(cls):
        try:
            return meta.explicit_kwargs[prop]
        except KeyError:
            pass
    return default


def data_aliases(cls, extra=None):
    """
    Return the data_aliases meta attribute for the given class.

    Extra is an optional dictionary with additional data alias definitions.
    """
    out = {}
    for base in reversed(cls.__bases__):
        try:
            out.update(base.meta.data_aliases)
        except AttributeError:
            continue

    out.update(extra or ())

    # Remove null aliases
    for k, v in list(out.items()):
        if v is None:
            del out[k]
    return out


def merge_with_ellipsis(xs: Iterable[T], extra: Sequence[T]) -> Iterator[T]:
    """
    Yield xs, but extra whenever encounters an ellipsis in xs.
    """
    for x in xs:
        if x is ...:
            yield from extra
        else:
            yield x
