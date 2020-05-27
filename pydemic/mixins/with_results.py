from abc import ABC
from collections import defaultdict
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

    _meta: "Meta"
    iter: int
    disease_params: Any
    region: Region
    get_param: callable
    RESULT_REGION_KEYS = ("population", "age_distribution", "age_pyramid")
    RESULT_DATES_KEYS = ("start", "end", "peak")

    @property
    def results(self) -> Results:
        return Results(self)

    def __init__(self):
        self._results_cache = defaultdict(dict)
        self._results_dirty_check = None

    def __getitem__(self, item):
        raise NotImplementedError

    def get_result_keys_data(self):
        """
        Yield keys for the result["data"] dict.
        """
        yield from self._meta.variables
        yield "cases"
        yield "attack_rate"

    def get_result_value_data(self, key):
        """
        Return value for model.result["data.<key>"] queries.
        """
        if key == "attack_rate":
            population = self["population:initial"]
            return (population - self["susceptible:final"]) / population
        return self[f"{key}:final"]

    def get_result_keys_params(self):
        """
        Yield keys for the result["params"] dict.
        """
        yield from self._meta.params.primary

    def get_result_value_params(self, key):
        """
        Return value for model.result["params.<key>"] queries.
        """
        return self.get_param(key)

    def get_result_keys_disease(self):
        """
        Yield keys for the result["disease"] dict.
        """
        yield from self.disease_params

    def get_result_value_disease(self, key):
        """
        Return value for model.result["disease.<key>"] queries.
        """
        return getattr(self.disease_params, key)

    def get_result_keys_region(self):
        """
        Yield keys for the result["region"] dict.
        """
        if self.region is None:
            return
        yield from self.RESULT_REGION_KEYS

    def get_result_value_region(self, key):
        """
        Handle model.result["region.*"] queries.
        """
        if key in self.RESULT_REGION_KEYS:
            return getattr(self, key)
        return KeyError(key)

    def get_result_keys_dates(self):
        """
        Yield keys for the result["dates"] dict.
        """
        yield from self.RESULT_DATES_KEYS

    def get_result_value_dates(self, key):
        """
        Return value for model.result["dates.<key>"] queries.
        """
        if key == "start":
            return self.to_date(0)
        elif key == "end":
            return self.date
        elif key == "peak":
            return self["infectious:peak-date"]
        else:
            raise KeyError(key)
