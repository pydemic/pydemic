# flake8: noqa
from mundi import Region

# noinspection PyUnresolvedReferences
from mundi_demography import age_distribution, age_pyramid, population

# noinspection PyUnresolvedReferences
from mundi_healthcare import (
    hospital_capacity,
    hospital_capacity_public,
    icu_capacity,
    icu_capacity_public,
)
from .base import RegionProperty
from .pydemic_property import PydemicProperty
from .pyplot_property import PyplotProperty


class RegionT(Region):
    plot: PyplotProperty
    pydemic: PydemicProperty


Region.pydemic = property(PydemicProperty)
Region.plot = property(PyplotProperty)
