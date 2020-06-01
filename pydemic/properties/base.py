import weakref
from abc import ABC, ABCMeta
from typing import Any

from ..logging import log


class PropertyMeta(ABCMeta):
    """
    Metaclass for property objects.

    Makes class behaves like a descriptor.
    """

    def __get__(cls, instance, kind=None):
        if instance is None:
            return cls
        else:
            return cls(instance)

    def __set_name__(cls, owner, name):
        cls.__dict__.setdefault("name", name)

    def as_property(cls):
        """
        Explicitly asks for a property interface in class declaration.

        >>> class Foo:
        ...     prop: Property = Property.as_property()
        """
        return property(cls)

    def patch(cls, klass: type = None, verbose=True):
        """
        Patch class with all public and private members of class.

        Dunder methods and attributes are omitted.

        Examples:
            >>> @Property.patch()
            ... class Mixin:
            ...     def method1(self):
            ...         pass
            ...     def method2(self):
            ...         pass
        """
        if klass is None:
            return lambda kind: cls.patch(kind)

        methods = set()
        for k, v in klass.__dict__.items():
            cls.add_method(v, k, verbose=False)
            methods.add(k)

        if verbose:
            cls_name = cls.__name__
            methods = ", ".join(map(repr, methods))
            log.info(f"{cls_name} patched with the {methods} methods")

        return klass

    def add_method(cls, func=None, name=None, verbose=True):
        """
        Patch class to include the given function as method. This is useful
        under interactive workloads until code is moved back to upstream.

        Examples:
            >>> @Property.add_method()
            ... def new_method(self, *args, **kwargs):
            ...     print("Property now as a .new_method method.")
        """
        if func is None:
            return lambda fn: cls.add_method(fn, name)

        if name is None:
            name = func.__name__
        setattr(cls, name, func)

        if verbose:
            cls_name = cls.__name__
            log.info(f"{cls_name} patched with a '{name}' method")

        return func


class BaseProperty(ABC, metaclass=PropertyMeta):
    """
    Base class for the several property accessors used in the Pydemic API.
    """

    __slots__ = ()
    _object: Any
    name: str

    def __repr__(self):
        try:
            name = self.name
        except AttributeError:
            name = type(self).__name__.lower()
            if name.endswith("property"):
                name = name[:8]

        cls = type(self._object).__name__
        return f"<{cls}.{name} property>"

    def __getstate__(self):
        return self._object

    def __setstate__(self, state):
        # noinspection PyArgumentList
        self.__init__(state)

    def __eq__(self, other):
        if type(self) is type(other):
            other: BaseProperty
            same_name = getattr(self, "name", None) == getattr(other, "name", None)
            return same_name and self._object == other._object
        return NotImplemented


class Property(BaseProperty):
    """
    Default base for pydemic properties namespaces.

    Uses weakrefs to avoid circular memory dependencies.
    """

    __slots__ = ("_ref",)
    _object = property(lambda self: self._ref())

    def __init__(self, obj):
        self._ref = weakref.ref(obj)


class StrongProperty(BaseProperty):
    """
    A version of Property for objects that do not support weak references.
    """

    __slots__ = ("_object",)

    def __init__(self, obj):
        self._object = obj
