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
    onset_to_hospitalization=5.0,
    icu_period=7.5,
    prob_hospitalization=0.18,
    prob_icu=0.05 / 0.18,
    prob_fatality=0.015 / 0.05,
    prob_no_hospitalization_fatality=0.30,
    prob_no_icu_fatality=1.00,
    case_fatality_rate=0.015,
    infection_fatality_rate=0.015 * 0.14,
    hospital_fatality_rate=0.05,
)
clinical = SimpleNamespace(DEFAULT=CLINICAL_DEFAULT)
