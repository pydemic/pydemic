from collections import MutableSequence
from typing import TYPE_CHECKING
from weakref import ref

if TYPE_CHECKING:
    from .model_group import ModelGroup


class ModelList(MutableSequence):
    """
    Implements the "models" attribute of a model group.

    It offers a list-like interface to the models included in a ModelGroup.
    """

    __slots__ = ("_data", "_group")

    group: "ModelGroup" = property(lambda self: self._group())
    _data: list

    def __init__(self, group: "ModelGroup", data: list):
        self._group = ref(group)
        self._data = data

    def __getitem__(self, idx):
        if isinstance(idx, str):
            for m in self._data:
                if m.name == idx:
                    return m
            else:
                raise KeyError(f"no model named {idx!r}")

        data = self._data[idx]
        if isinstance(data, list):
            cls = type(self._group)
            return cls(data)
        return data

    def __setitem__(self, idx, obj) -> None:
        old = self[idx]
        model_cls = self.group.kind
        if isinstance(old, type(self._group)):
            data = getattr(obj, "models", obj)
            if not all(isinstance(m, model_cls) for m in data):
                raise self._insert_type_error()
            self._data[idx] = data
        elif isinstance(obj, model_cls):
            self._data[idx] = obj
        else:
            raise self._insert_type_error()

    def __delitem__(self, i: int) -> None:
        del self._data[i]

    def __len__(self) -> int:
        return len(self._data)

    def insert(self, index: int, object) -> None:
        if not isinstance(object, self.group.kind):
            raise self._insert_type_error()
        self._data[index] = object

    def _insert_type_error(self):
        return TypeError("can only insert model instances to model group")
