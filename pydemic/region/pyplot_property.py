from pydemic.utils import trim_weeks
from .base import RegionProperty, cached
from .. import plot as plt


class PyplotProperty(RegionProperty):
    __slots__ = ()

    @cached
    def weekday_rate(self, disease=None, **kwargs):
        """
        Show rate of notification by weekday.
        """
        data = self.region.pydemic.epidemic_curves(disease, new_cases=True)
        data = trim_weeks(data)
        return plt.weekday_rates(data, **kwargs)
