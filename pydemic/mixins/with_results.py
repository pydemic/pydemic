from abc import ABC

from .results import Results


class WithSummaryMixin(ABC):
    """
    Subclasses that adopt this mixin gain a "summary" attribute that expose
    useful information about its owner.

    In the context of simulation models, it shows useful information about the
    evolution of the epidemic.
    """

    results: Results

    def __init__(self):
        self.results = Results(self)

    def get_result_dates(self, key):
        if key.startswith("peak_"):
            raise NotImplementedError
        else:
            fmt = key.replace(".", "__")
            try:
                method = getattr(self, f"get_result_dates__{fmt}")
            except AttributeError:
                return ValueError(f'invalid result key, "dates.{key}"')
            else:
                return method()
