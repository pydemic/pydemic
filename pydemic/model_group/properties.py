import operator
from typing import TYPE_CHECKING

from .utils import map_models, map_method, model_group_method, group

if TYPE_CHECKING:
    from .model_group import ModelGroup


class ModelGroupProp:
    """
    Base class for all model group properties.
    """

    group: "ModelGroup"
    prop_name: str = None
    MODEL_METHODS = frozenset()
    PROP_METHODS = frozenset()

    def __init__(self, group):
        self.group = group

    def __getitem__(self, item):
        models = self.iter_items(item)
        return map_models(operator.itemgetter(item), models)

    def __getattr__(self, item):
        from .model_group import ModelGroup

        if item in self.MODEL_METHODS:

            def method(*args, **kwargs):
                return ModelGroup(fn(*args, **kwargs) for fn in self.iter_attr(item))

            method.__name__ = item
            return method

        if item in self.PROP_METHODS:

            def method(*args, **kwargs):
                return [fn(*args, **kwargs) for fn in self.iter_attr(item)]

            method.__name__ = item
            return method

        return list(self.iter_attr(item))

    def __dir__(self):
        try:
            prop = next(self.iter_pros())
            prop_attrs = dir(prop)
        except IndexError:
            prop_attrs = ()
        return list({*super().__dir__(), *prop_attrs})

    def iter_props(self):
        """
        Iterate over each accessor.
        """
        prop_name = self.prop_name
        for m in self.group:
            yield getattr(m, prop_name)

    def iter_items(self, item):
        """
        Iterate over each accessor, selecting the given item.
        """
        for prop in self.iter_props():
            yield prop[item]

    def iter_attr(self, attr):
        """
        Iterate over each accessor, selecting the given attribute.
        """
        for prop in self.iter_props():
            yield getattr(prop, attr)

    def iter_method(self, *args, **kwargs):
        """
        Iterate over each accessor, executing the given method forwarding
        the passed arguments.
        """
        method, *args = args
        for fn in self.iter_attr(method):
            yield fn(*args, **kwargs)


class ModelGroupInfo(ModelGroupProp):
    """
    Implements the "info" attribute of model groups.
    """

    prop_name = "info"


class ModelGroupResults(ModelGroupProp):
    """
    Implements the "results" attribute of model groups.
    """

    prop_name = "results"


class ModelGroupClinical(ModelGroupProp):
    """
    Implements the "clinical" attribute of model groups.
    """

    prop_name = "clinical"
    MODEL_METHODS = {"clinical_model", "crude_model", "delay_model", "overflow_model"}

    def __call__(self, *args, **kwargs):
        return group(map_method("clinical", self.group, *args, **kwargs))

    crude_model = model_group_method("clinical.crude_model", out=group)
    delay_model = model_group_method("clinical.delay_model", out=group)
    overflow_model = model_group_method("clinical.overflow_model", out=group)
