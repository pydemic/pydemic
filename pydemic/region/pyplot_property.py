from .base import RegionProperty, cached


class PyplotProperty(RegionProperty):
    __slots__ = ()

    @cached
    def weekday_rate(self, disease=None, **kwargs):
        """
        Show rate of notification by weekday.
        """
        data = self.region.pydemic.epidemic_curves(disease, new_cases=True)
        data = trim_weeks(data)
        return plot_weekday_rate(data, **kwargs)
