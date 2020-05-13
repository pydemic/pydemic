import sidekick as sk
from sidekick import placeholder as _


class WrappedParams:
    """
    A wrapper object that exposes a disease as a param namespace.
    """

    gamma = sk.property(1 / _.infectious_period)
    sigma = sk.property(1 / _.incubation_period)

    Qs = sk.alias("prob_symptoms")
    Qsv = sk.alias("prob_severe")
    Qcr = sk.alias("prob_critical")
    CFR = sk.alias("case_fatality_rate")
    IFR = sk.alias("infection_fatality_rate")
    HFR = sk.alias("hospital_fatality_rate")
    ICUFR = sk.alias("icu_fatality_rate")

    __slots__ = ("_wrapped", "_args", "_kwargs", "_cache")

    def __init__(self, wrapped, *args, **kwargs):
        self._wrapped = wrapped
        self._args = args
        self._kwargs = kwargs
        self._cache = {}

    def __getitem__(self, item):
        try:
            return self._fetch_param(item)
        except ValueError:
            raise KeyError(item)

    def __getattr__(self, item):
        try:
            return self._fetch_param(item)
        except ValueError:
            raise AttributeError(item)

    def _fetch_param(self, name):
        """
        Obtain parameter with the given name and save result in cache.
        """
        try:
            return self._cache[name]
        except KeyError:
            try:
                value = getattr(self._wrapped, name)
            except AttributeError:
                raise ValueError(f"invalid parameter: {name!r}")
            if callable(value):
                value = value(*self._args, **self._kwargs)
            self._cache[name] = value
            return value
