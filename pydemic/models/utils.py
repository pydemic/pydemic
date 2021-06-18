from numbers import Real
from typing import Optional, Union, Tuple

import mundi
import pandas as pd

INIT_KEYWORDS = frozenset(["region", "population", "age_distribution", "age_pyramid"])


def extract_demography(
    region: Union[str, mundi.Region, None] = None,
    population: Union[str, mundi.Region, int, None] = None,
    age_distribution: Union[str, mundi.Region, pd.Series, None] = None,
    age_pyramid: Union[str, mundi.Region, pd.DataFrame, None] = None,
) -> Tuple[Optional[Real], Optional[pd.Series], Optional[pd.DataFrame]]:
    """
    Extract (population, age_distribution, age_pyramid) demographic parameters.
    """

    if age_distribution is not None and age_pyramid is not None:
        msg = "cannot set age_pyramid and age_distribution simultaneously"
        raise ValueError(msg)

    if region is not None:
        region = mundi.region(region)

    # Simple case where no demographic parameter is given
    if population is None and age_distribution is None and age_pyramid is None:
        population = fallback_to_region(region, "population")
        age_distribution = fallback_to_region(region, "age_distribution")
        age_pyramid = fallback_to_region(region, "age_pyramid")
        return population, age_distribution, age_pyramid

    # If age pyramid is a dataframe, it determines all other parameters
    if isinstance(age_pyramid, pd.DataFrame):
        if population is not None:
            raise TypeError("age_pyramid already fixes population")
        if age_distribution is not None:
            raise TypeError("age_pyramid already fixes age_distribution")
        age_distribution = age_pyramid["female"] + age_pyramid["male"]
        age_distribution.index.name = "age"
        population = age_distribution.sum()
        return population, age_distribution, age_pyramid

    # Similarly, age_distribution may be given to determine population
    if isinstance(age_distribution, (pd.Series, pd.DataFrame)):
        if age_pyramid is not None:
            raise TypeError("cannot specify age_pyramid if age_distribution is explicit")
        population = age_distribution.values.sum()
        return population, age_distribution, age_pyramid

    # If we only have population and region, population is used to adjust the regional
    # distribution. The case in which population is None is already covered
    if age_pyramid is None and age_distribution is None:
        if region is None:
            return population, age_distribution, age_pyramid

        if region.age_pyramid is not None:
            ratio = population / region.population
            age_pyramid = region.age_pyramid * ratio
            age_distribution = region.age_distribution * ratio
            return population, age_distribution, age_pyramid

        if region.age_distribution is not None:
            ratio = population / region.population
            return population, region.age_distribution * ratio, None

        return population, None, None

    # Give up in all other cases?
    raise NotImplementedError


def fallback_to_region(
    value: Union[str, mundi.Region, pd.DataFrame, None], attr: str
) -> Union[Real, pd.Series, pd.DataFrame, None]:
    """
    Return either value or the given attribute of a region, if value is a
    string or Region instance.
    """
    if value is None:
        return None
    elif isinstance(value, (str, mundi.Region)):
        region = mundi.region(value)
        value = getattr(region, attr)
    elif not hasattr(value, "copy"):
        return value
    try:
        cp = value.copy()
        if value.isna().values.all():
            return None
        return cp
    except AttributeError:
        return value
