from types import SimpleNamespace

from .param import Params


class ClinicalParams(Params):
    pass


#
# Clinical parameters
#
CLINICAL_DEFAULT = Params(
    "Default",
    hospitalization_period=7.0,
    icu_period=7.5,
    severe_delay=5.0,
    hospitalization_overflow_bias=0.25,
    critical_delay=7.0,
    prob_severe=0.18,
    prob_critical=0.05,
    prob_fatality=0.015 / 0.05,
    prob_no_hospitalization_fatality=0.30,
    prob_no_icu_fatality=1.00,
    case_fatality_ratio=0.015,
    infection_fatality_ratio=0.015 * 0.14,
    hospital_fatality_ratio=0.05,
)
clinical = SimpleNamespace(DEFAULT=CLINICAL_DEFAULT)
