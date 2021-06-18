from ..types import ComputedProxyDict
from pydemic.types import inverse, alias


class DiseaseParams(ComputedProxyDict):
    """
    A wrapper for disease params.
    """

    gamma: float = inverse("infectious_period")
    sigma: float = inverse("incubation_period")

    Qs: float = alias("prob_symptoms")
    Qsv: float = alias("prob_severe")
    Qcr: float = alias("prob_critical")
    CFR: float = alias("case_fatality_ratio")
    IFR: float = alias("infection_fatality_ratio")
    HFR: float = alias("hospital_fatality_ratio")
    ICUFR: float = alias("icu_fatality_ratio")

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)
