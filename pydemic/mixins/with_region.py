from typing import Optional, TYPE_CHECKING

import pandas as pd

import mundi
from ..utils import extract_keys

INIT_KEYWORDS = frozenset(["region", "population", "age_distribution", "age_pyramid"])

if TYPE_CHECKING:
    from ..region import RegionT  # noqa: F401


class WithRegionDemography:
    """
    Models with a .region attribute and the possibility of overriding basic
    demographic parameters.
    """

    region: Optional["RegionT"]
    population: float
    age_distribution: Optional[pd.Series]
    age_pyramid: Optional[pd.DataFrame]

    def _init_from_dict(self, kwargs, drop=True):
        """
        Init from dictionary removing used keys.
        """
        opts = extract_keys(INIT_KEYWORDS, kwargs, drop=drop)
        self.__init(**opts)
        return opts

    def __init__(self, **kwargs):
        self.__init(**kwargs)

    def __init(self, region=None, population=None, age_distribution=None, age_pyramid=None):

        if age_distribution is not None and age_pyramid is not None:
            msg = "cannot set age_pyramid and age_distribution simultaneously"
            raise ValueError(msg)

        # Set region
        if region is not None:
            self.region = mundi.region(region)
            population = population or self.region.population
        elif not hasattr(self, "region"):
            self.region = None

        # Set age_pyramid
        if age_pyramid is not None:
            self.age_pyramid = fallback_to_region(age_pyramid, "age_pyramid")
        elif hasattr(self, "age_pyramid"):
            pass
        elif self.region is not None:
            self.age_pyramid = fallback_to_region(self.region, "age_pyramid")
        else:
            self.age_pyramid = None

        # Set age_distribution
        if age_distribution is not None:
            value = fallback_to_region(age_distribution, "age_distribution")
            self.age_distribution = value
        elif hasattr(self, "age_distribution"):
            pass
        elif self.age_pyramid is not None:
            self.age_distribution = self.age_pyramid.sum(1)
            self.age_distribution.name = "age_distribution"
            self.age_distribution.index.names = ["age"]
        elif self.region is not None:
            self.age_distribution = fallback_to_region(self.region, "age_distribution")
        else:
            self.age_distribution = None

        # Set population and fix age_distribution and age_pyramid, if necessary
        if hasattr(self, "population"):
            pass
        elif population is not None:
            population = fallback_to_region(population, "population")
            self.population = population

            if self.age_distribution is not None:
                ratio = population / self.age_distribution.sum()
                if ratio != 1:
                    self.age_distribution *= ratio

            if self.age_pyramid is not None:
                ratio = population / self.age_pyramid.sum().sum()
                if ratio != 1:
                    self.age_pyramid *= ratio
        elif self.age_distribution is not None:
            self.population = self.age_distribution.sum()
        elif self.region is not None:
            self.population = self.region.population
        else:
            self.population = 1_000_000


def fallback_to_region(value, attr):
    """
    Return either value or the given attribute of a region, if value is a
    string or Region instance.
    """
    if value is None:
        return None
    elif isinstance(value, (str, mundi.Region)):
        region = mundi.region(value)
        value = getattr(region, attr)
    try:
        cp = value.copy()
        if value.isna().values.all():
            return None
        return cp
    except AttributeError:
        return value
