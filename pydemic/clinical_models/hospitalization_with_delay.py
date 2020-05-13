from .crude_FR import CrudeFR
from .utils import delayed, delayed_with_discharge
from ..params import clinical
from ..utils import param_property


class HospitalizationWithDelay(CrudeFR):
    """
    Model in which infected can become hospitalized and suffer a constant
    hospitalization fatality rate.

    Attributes:
        severe_delay (float):
            Duration between symptom onset to hospitalization.
        critical_delay (float):
            Duration between symptom onset to ICU treatment.
    """

    params = clinical.DEFAULT

    # Primary parameters
    severe_delay = param_property(default=0.0)
    critical_delay = param_property(default=0.0)

    #
    # Data methods
    #
    def get_data_deaths(self):
        K = self.growth_factor
        try:
            critical = self["critical_cases"]
        except KeyError:
            return delayed(self["severe_cases"] * self.HFR, self.hospitalization_period, K)
        else:
            return delayed(critical * self.ICUFR, self.icu_period, K)

    def get_data_critical(self):
        data = self["critical_cases"]
        return delayed_with_discharge(data, 0, self.icu_period, self.K)

    def get_data_severe_cases(self):
        K = self.growth_factor
        return delayed(self["cases"] * self.Qsv, self.severe_delay, K)

    def get_data_critical_cases(self):
        K = self.growth_factor
        values = self["severe_cases"] * (self.Qcr / self.Qsv)
        delay = self.severe_delay - self.critical_delay
        return delayed(values, delay, K)
