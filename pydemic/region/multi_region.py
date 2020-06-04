from typing import Optional, FrozenSet, Union, Any

import numpy as np
import pandas as pd
import sidekick as sk

from mundi import Region, region
from .pydemic_property import PydemicProperty
from .pyplot_property import PyplotProperty
from ..diseases import disease as get_disease
from ..properties.decorators import cached

AGGREGATE_METHOD = {
    "population": "sum",
    "age_distribution": "sum",
    "age_pyramid": "sum",
    "icu_capacity": "sum",
    "hospital_capacity": "sum",
    "icu_capacity_public": "sum",
    "hospital_capacity_public": "sum",
}


class CompositePydemicProperty(PydemicProperty):
    region: "CompositeRegion"

    @cached(ttl="epidemic_curve")
    def epidemic_curve(self, disease=None, **kwargs) -> pd.DataFrame:
        fn = get_disease(disease).epidemic_curve
        curves = (fn(r, **kwargs) for r in self.region.regions)
        return self.region.aggregate_values(curves, "sum")


class CompositePyplotProperty(PyplotProperty):
    region: "CompositeRegion"


class CompositeRegion(Region):
    """
    An aggregate of regions.
    """

    name: str
    regions: FrozenSet[Region]
    _keys: FrozenSet[str]

    pydemic = property(CompositePydemicProperty)
    plot = property(CompositePyplotProperty)

    @property
    def id(self):
        try:
            return self.__dict__["id"]
        except KeyError:
            out = "|".join(sorted(map(str, self.regions)))
            self.__dict__["id"] = out
        return out

    def __new__(cls, sub_regions, **kwargs):
        sub_regions = frozenset(map(region, sub_regions))
        if not sub_regions:
            raise ValueError("cannot start with empty list of regions")
        new = object.__new__(cls)
        new.__dict__["regions"] = sub_regions
        new.__dict__["_keys"] = frozenset(kwargs)
        new.__dict__.update(**kwargs)
        return new

    def __hash__(self):
        return hash(self.__getstate__())

    def __getstate__(self):
        args = ((k, self[k]) for k in self._keys)
        return self.regions, args

    def __setstate__(self, state):
        state = self.__dict__
        regions, args = state
        state["regions"] = regions
        for k, v in args:
            state[k] = v

    def __str__(self):
        regions = [r.id for r in self.regions]
        if len(regions) > 5:
            regions = regions[:5] + "..."
        regions = ", ".join(regions)

        return (
            f"Composite Region\n"
            f"id       : {self.id}\n"
            f"name     : {self.name}\n"
            f"type     : {self.type}\n"
            f"subtype  : {self.subtype}\n"
            f"regions  : {regions}"
        )

    def __eq__(self, other):
        if isinstance(other, CompositeRegion):
            keys = self._keys.union(other._keys)
            equal_attrs = (force_bool(self[k] == other[k]) for k in keys)
            return self.regions == other.regions and all(equal_attrs)
        return NotImplemented

    def __repr__(self):
        cls = type(self).__name__
        return f"{cls}({self.regions, !r})"

    def _get_field(self, key):
        method = AGGREGATE_METHOD.get(key, "single")
        return self.aggregate(key, method)

    def aggregate(self, field, by):
        """
        Aggregate values with function.
        """
        values = (region._get_field(field) for region in self.regions)
        return self.aggregate_values(values, by)

    def aggregate_values(self, values, by):
        """
        Given a sequence or iterable with one value per region, aggregate
        values using the given method.
        """
        if by == "sum":
            value, values = sk.uncons(values)
            try:
                value = value.copy()
            except AttributeError:
                return sum(values, value)
            else:
                for v in values:
                    value += v
                return value
        elif by == "sum-pop":
            populations = (region.population for region in self.regions)
            values = (x * pop for x, pop in zip(values, populations))
            return self.aggregate_values(values, "sum")
        elif by == "mean":
            return self.aggregate(values, "sum") / len(self.regions)
        elif by == "mean-pop":
            return self.aggregate(values, "sum-pop") / self.population
        elif by == "single":
            value, *error = set(values)
            if error:
                raise ValueError("regions do not share the same value")
            return value
        else:
            raise ValueError(f"invalid aggregation method: {by}")

    #
    # Hierarchies
    #
    @property
    def parent(self) -> Optional["Region"]:
        """
        Common parent to all regions.
        """
        parent, *error = set(region.parent for region in self.regions)
        if error:
            raise ValueError("regions do not share a common parent.")
        return parent

    def parents(self, dataframe=False):
        if dataframe:
            raise NotImplementedError
        parent = self.parent
        return [parent, *parent.parents()]

    def children(self, dataframe=False, deep=False, which="both"):
        """
        Return list of children.
        """
        if dataframe:
            raise NotImplementedError

        children = {*self.regions}
        if deep:
            for region in self.regions:
                children.update(region.children(deep=True, which=which))
        return list(children)


def force_bool(x: Any) -> bool:
    """
    Similar to bool(x), but call the .all() method until object reduces to a
    real boolean.

    This is useful to test numpy arrays and pandas data structures for equality.
    """
    x: Union[bool, np.ndarray] = bool(x)
    if x is True or x is False:
        return x
    return force_bool(x.all())
