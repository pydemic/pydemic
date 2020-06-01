import sidekick as sk
from sidekick import placeholder as _

from pydemic.params import WrappedParams


class DiseaseParams(WrappedParams):
    """
    A wrapper for disease params.
    """

    gamma: float = sk.property(1 / _.infectious_period)
    sigma: float = sk.property(1 / _.incubation_period)

    Qs: float = sk.alias("prob_symptoms")
    Qsv: float = sk.alias("prob_severe")
    Qcr: float = sk.alias("prob_critical")
    CFR: float = sk.alias("case_fatality_ratio")
    IFR: float = sk.alias("infection_fatality_ratio")
    HFR: float = sk.alias("hospital_fatality_ratio")
    ICUFR: float = sk.alias("icu_fatality_ratio")
