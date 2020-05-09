import sidekick as sk

from .crude_FR import CrudeFR
from .utils import delayed, delayed_with_discharge
from ..params import clinical
from ..utils import param_property, param_alias

healthcare = sk.import_later("mundi_healthcare")


class HospitalizationWithDelay(CrudeFR):
    """
    Model in which infected can become hospitalized and suffer a constant
    hospitalization fatality rate.
    """

    params = clinical.DEFAULT

    # Primary parameters
    onset_to_hospitalization = param_property()
    hospital_fatality_rate = param_property()

    # FIXME: hardcoded values should become proper parameters
    hospitalization_to_icu = 4.0
    icu_period = 7.5

    # Aliases
    qh = param_alias("prob_hospitalization")
    qc = 0.25
    qc_overflow_bias = 1.5
    HFR = param_alias("hospital_fatality_rate")

    #
    # Data methods
    #
    def get_data_deaths(self):
        return delayed(self["severe_cases"] * self.HFR, self.hospitalization_period)

    def get_data_critical(self):
        data = self["critical_cases"]
        return delayed_with_discharge(data, 0, self.icu_period)

    def get_data_severe_cases(self):
        return delayed(self["cases"] * self.qh, self.onset_to_hospitalization)

    def get_data_critical_cases(self):
        return delayed(self["severe_cases"] * self.qc, self.hospitalization_to_icu)
