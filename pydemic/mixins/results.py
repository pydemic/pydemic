from .base_attr import BaseAttr


class Results(BaseAttr):
    """
    Results objects store dynamic information about a simulation.

    Results data is not necessarily static throughout the simulation. It can
    include values like total death toll, number of infected people, etc. It also
    distinguishes from information stored in the model mapping interface in
    that it does not include time series.

    Most information available as ``m[<param>]`` will also be available as
    ``m.results[<param>]``. While the first typically include the whole time
    series for the object, the second typically correspond to the last value
    in the time series.
    """

    __slots__ = ()
    _method_namespace = "results"

    @property
    def _cache(self):
        model = self.model
        if model.iter != model._results_dirty_check:
            model._results_cache.clear()
            model._results_dirty_check = model.iter
        return model._results_cache
