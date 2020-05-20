import pandas as pd


class Results:
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

    __slots__ = ("owner",)

    def __init__(self, obj):
        self.owner = obj

    def __getitem__(self, item):
        prefix, _, tail = item.partition(".")

        try:
            return self.owner._results_data[item]
        except KeyError:
            pass

        if not tail:
            return self.owner[f"{item}:final"]
        else:
            name = f"get_result_{prefix}"
            if hasattr(self.owner, name):
                method = getattr(self.owner, name)
                return method(tail)
            else:
                cls = type(self.owner).__name__
                raise KeyError(f"{cls} object has no {item!r} result key.")

    def to_frame(self, which="summary") -> pd.DataFrame:
        """
        Expose information about object as a dataframe.
        """
        raise NotImplementedError

    def _html_repr_(self):
        return self.to_frame()._html_repr_()
