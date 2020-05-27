from abc import ABC
from collections import defaultdict
from typing import TYPE_CHECKING

from .info import Info

if TYPE_CHECKING:
    from ..models import Model  # noqa: F401


class WithInfoMixin(ABC):
    """
    Subclasses that adopt this mixin gain a "info" attribute that expose
    useful information about its owner.

    In the context of simulation models, it shows useful static information
    about the simulation.
    """

    INFO_DEMOGRAPHY_KEYS = ("population", "age_distribution", "age_pyramid")
    INFO_REGION_KEYS = (
        "population",
        "age_distribution",
        "age_pyramid",
        "icu_capacity",
        "hospital_capacity",
    )

    def __init__(self):
        self._info_cache = defaultdict(dict)

    @property
    def info(self) -> Info:
        return Info(self)

    def get_info_keys_demography(self):
        """
        Yield keys for the info["demography"] dict.
        """
        yield from self.INFO_DEMOGRAPHY_KEYS

    def get_info_value_demography(self: "Model", key):
        """
        Return value for model.info["demography.<key>"] queries.
        """
        if key == "population":
            return self.data.iloc[0].sum()
        elif key == "age_distribution":
            return self.age_distribution
        elif key == "age_pyramid":
            return self.age_pyramid
        elif key.startswith("population("):
            _, _, cmd = key.partition("(")
            cmd = cmd.rstrip(")")
            return get_population_fraction(cmd, self.age_distribution)
        else:
            raise ValueError(f"unknown argument: {key!r}")

    def get_info_keys_region(self: "Model"):
        """
        Yield keys for the result["region"] dict.
        """
        if self.region is None:
            return
        yield from self.INFO_REGION_KEYS

    def get_info_value_region(self: "Model", key):
        """
        Handle model.result["region.*"] queries.
        """
        if key in self.INFO_REGION_KEYS:
            return getattr(self.region, key)
        return KeyError(key)

    def get_info_keys_disease(self: "Model"):
        """
        Yield keys for the result["disease"] dict.
        """
        yield from self.disease_params

    def get_info_value_disease(self: "Model", key):
        """
        Return value for model.result["disease.<key>"] queries.
        """
        return getattr(self.disease_params, key)


def get_population_fraction(cmd, distribution):
    """
    Given a command like 60+, <10, 50-59, etc, return the fraction of population
    in the given interval.
    """

    total = distribution.sum()
    if cmd.endwith("+"):
        pos = int(cmd[:-1])
        return distribution.loc[pos:].sum() / total
    elif cmd.startswith("<"):
        pos = int(cmd[1:])
        return distribution.loc[:pos].sum() / total
    else:
        a, b = map(int, cmd.split("-"))
        b += 1
        return distribution.loc[a:b].sum() / total
