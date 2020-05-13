import sidekick as sk
from sidekick import placeholder as _

from .model import ClinicalObserverModel
from .utils import delayed_with_discharge
from ..params import clinical
from ..utils import param_property, param_alias


class CrudeFR(ClinicalObserverModel):
    """
    Model in which infected can become hospitalized and suffer a constant
    hospitalization fatality rate.

    Attributes:
        prob_severe (float, alias: Qsv):
            Probability that regular cases become severe (cases that require
            hospitalization).
        prob_critical (float, alias: Qcr):
            Probability that regular cases become critical (require ICU).
        hospitalization_period:
            Average duration of hospitalizations.
        icu_period:
            Average duration of ICU treatment.
        hospital_fatality_rate (float, alias: HFR):
            Fraction of deaths for patients that go to hospitalization.
        icu_fatality_rate (float, alias: ICUFR):
            Fraction of deaths for patients that go to ICU treatment.
    """

    params = clinical.DEFAULT
    growth_factor = sk.alias("K")

    # Primary parameters
    prob_severe: float = param_property(default=0.0)
    prob_critical: float = param_property(default=0.0)

    hospitalization_period: float = param_property(default=0.0)
    icu_period: float = param_property(default=0.0)

    # Aliases
    Qsv: float = param_alias("prob_severe")
    Qcr: float = param_alias("prob_critical")

    # Properties
    hospital_fatality_rate = sk.property(_.CFR / _.Qsv)
    HFR = sk.alias("hospital_fatality_rate")

    icu_fatality_rate = sk.property(_.CFR / _.Qcr)
    ICUFR = sk.alias("icu_fatality_rate")

    prob_aggravate_to_icu = sk.property(_.Qcr / _.Qsv)

    # Cumulative series
    def get_data_deaths(self):
        return self["cases"] * self.CFR

    def get_data_severe(self):
        data = self["severe_cases"]
        K = self.growth_factor
        return delayed_with_discharge(data, 0, self.hospitalization_period, K)

    def get_data_severe_cases(self):
        return self["cases"] * self.Qsv

    def get_data_critical_cases(self):
        return self["cases"] * self.Qcr
