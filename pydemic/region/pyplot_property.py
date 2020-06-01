from .base import RegionProperty, cached
from .. import plot as plt
from ..utils import trim_weeks


class PyplotProperty(RegionProperty):
    __slots__ = ()

    @cached
    def weekday_rate(self, disease=None, **kwargs):
        """
        Show rate of notification by weekday.
        """
        data = self.region.pydemic.epidemic_curve(disease, diff=True)
        data = trim_weeks(data)
        return plt.weekday_rates(data, **kwargs)

    @cached
    def cases_and_deaths(self, disease=None, **kwargs):
        """
        Return a plot of cases and deaths
        """
        curves = self.region.pydemic.epidemic_curve(disease)
        kwargs.setdefault("tight_layout", True)
        return plt.cases_and_deaths(curves, **kwargs)
