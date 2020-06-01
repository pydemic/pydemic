import sidekick as sk
from sidekick import placeholder as _

from .model import ClinicalObserverModel
from .utils import delayed_with_discharge
from ..params import clinical
from ..utils import param_property, param_alias, sliced


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
        hospital_fatality_ratio (float, alias: HFR):
            Fraction of deaths for patients that go to hospitalization.
        icu_fatality_ratio (float, alias: ICUFR):
            Fraction of deaths for patients that go to ICU treatment.
    """

    params = clinical.DEFAULT

    # Primary parameters
    prob_severe: float = param_property(default=0.0)
    prob_critical: float = param_property(default=0.0)

    severe_period: float = param_property(default=0.0)
    critical_period: float = param_property(default=0.0)

    # Aliases
    Qsv: float = param_alias("prob_severe")
    Qcr: float = param_alias("prob_critical")

    # Properties
    hospital_fatality_ratio = sk.property(_.CFR / _.Qsv)
    HFR = sk.alias("hospital_fatality_ratio")

    icu_fatality_ratio = sk.property(_.CFR / _.Qcr)
    ICUFR = sk.alias("icu_fatality_ratio")

    prob_aggravate_to_icu = sk.property(_.Qcr / _.Qsv)

    # Cumulative series
    def get_data_deaths(self, idx):
        return self["cases", idx] * self.CFR

    def get_data_severe(self, idx):
        data = self["severe_cases"]
        K = max(self.K, 0)
        data = delayed_with_discharge(data, 0, self.severe_period, K, positive=True)
        return sliced(data, idx)

    def get_data_severe_cases(self, idx):
        return self["cases", idx] * self.Qsv

    def get_data_critical_cases(self, idx):
        return self["cases", idx] * self.Qcr
