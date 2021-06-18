from collections.abc import Sequence

import mundi
import sidekick.api as sk
from .epidemiology import Epidemiology
from .utils import delegate_map, delegate_mapping, concat_frame, agg_frame

info_method = lambda attr: delegate_mapping(attr, concat_frame)
sum_method = lambda attr: delegate_mapping(attr, agg_frame("sum"))
mean_method = lambda attr: delegate_mapping(attr, agg_frame("mean"))


class EpidemiologyGroup(Epidemiology, Sequence):
    """
    A collection of Epidemiology instances.

    It exposes an Epidemiology interface with aggregate values and methods that
    disaggregate time series as different columns of a data frame.
    """

    __slots__ = ("_data", "_keys")
    __len__ = sk.delegate_to("_data")
    __iter__ = sk.delegate_to("_data")
    __getitem__ = sk.delegate_to("_data")
    count = sk.delegate_to("_data")
    index = sk.delegate_to("_data")

    # Transformations
    trim_dates = delegate_map("trim_dates")
    moving_average = delegate_map("moving_average")
    clean = delegate_map("clean")

    # Information - gathers dataframe from each item
    confirmed_cases = sum_method("confirmed_cases")
    corrected_cases = sum_method("corrected_cases")
    confirmed_deaths = sum_method("confirmed_deaths")
    new_cases = sum_method("new_cases")
    population = sum_method("population")
    # ascertainment_rate = info_method('ascertainment_rate')
    # attack_rate = info_method('attack')
    # incidence_rate = info_method('incidence_rate')
    # point_prevalence = info_method('point_prevalence')
    # period_prevalence = info_method('period_prevalence')

    # Information - disaggregate data for each item
    each_confirmed_cases = info_method("confirmed_cases")
    each_corrected_cases = info_method("corrected_cases")
    each_confirmed_deaths = info_method("confirmed_deaths")
    each_new_cases = info_method("new_cases")
    each_population = info_method("population")
    each_ascertainment_rate = info_method("ascertainment_rate")
    each_attack_rate = info_method("attack")
    each_incidence_rate = info_method("incidence_rate")
    each_point_prevalence = info_method("point_prevalence")
    each_period_prevalence = info_method("period_prevalence")

    @classmethod
    def from_regions(cls, regions, **kwargs):
        """
        Initialize group from sequence of regions.
        """
        regions = map(mundi.region, regions)
        new = EpidemicCurve.from_region
        return cls({r.id: new(r, **kwargs) for r in regions})

    @classmethod
    def from_query(cls, *args, disease=None, params=None, **kwargs):
        """
        Initialize group from a mundi region query.
        """
        regions = mundi.regions_dataframe(*args, **kwargs)
        return cls.from_regions(regions.index, disease=disease, params=params)

    def __init__(self, curves):
        try:
            self._data = [*curves.values()]
            self._keys = [*curves.keys()]
        except AttributeError:
            self._data = [*curves]
            self._keys = range(len(self._data))
