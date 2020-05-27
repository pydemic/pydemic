from abc import ABC
from typing import TYPE_CHECKING, Any

from mundi import Region
from .results import Results

if TYPE_CHECKING:
    from .meta_info import Meta


class WithResultsMixin(ABC):
    """
    Subclasses that adopt this mixin gain a "summary" attribute that expose
    useful information about its owner.

    In the context of simulation models, it shows useful information about the
    evolution of the epidemic.
    """

    iter: int
    _meta: "Meta"
    get_param: callable
    disease_params: Any
    region: Region

    @property
    def results(self) -> Results:
        return Results(self)

    def __init__(self):
        self._results_cache = {}
        self._results_dirty_check = None

    def get_result_data(self, key=None):
        """
        Handle model.result["data.*"] queries.
        """

        if key is None:
            keys = [*self._meta.variables, "cases", "attack_rate", *extra_keys(self, "data")]
            return {k: self.get_result_data(k) for k in keys}

        if key == "attack_rate":
            population = self["population:initial"]
            return (population - self["susceptible:final"]) / population

        return self[f"{key}:final"]

    def get_result_params(self, key=None):
        """
        Handle model.result["param.*"] queries.
        """
        if key is None:
            keys = [*self._meta.params.primary, *extra_keys(self, "param")]
            return {k: self.get_result_params(k) for k in keys}

        return self.get_param(key)

    def get_result_disease(self, key=None):
        """
        Handle model.result["disease.*"] queries.
        """
        if key is None:
            keys = [*self.disease_params, *extra_keys(self, "disease")]
            return {k: self.get_result_disease(k) for k in keys}

        return getattr(self.disease_params, key)

    def get_result_region(self, key=None):
        """
        Handle model.result["region.*"] queries.
        """
        keys = ("population", "age_distribution", "age_pyramid")
        if key is None and self.region is None:
            return None
        elif key is None:
            keys = (*keys, *extra_keys(self, "region"))
            return {k: self.get_result_region(k) for k in keys}

        if key in keys:
            return getattr(self, key)
        return KeyError(key)

    def get_result_dates(self, key):
        """
        Handle model.result["disease.*"] queries.
        """

        keys = ("start", "end", "peak")
        if key is None:
            keys = (*keys, *extra_keys(self, "dates"))
            return {k: self.get_result_dates(k) for k in keys}
        elif key == "start":
            return self.to_date(0)
        elif key == "end":
            return self.date
        elif key == "peak":
            return self["infectious:peak-date"]
        else:
            raise KeyError(key)


def extra_keys(model, name):
    """
    Collect extra keys implemented as _get_result_<name>__<key> methods.
    """

    prefix = f"_get_result_{name}__"
    n = len(prefix)

    for attr in dir(model):
        if attr.startswith(prefix):
            yield attr[n:].replace("__", ".")
