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
    severe_delay: float = param_property(default=0.0)
    critical_delay: float = param_property(default=0.0)

    #
    # Data methods
    #
    def get_data_deaths(self, idx):
        K = max(self.K, 0)
        try:
            critical = self["critical_cases", idx]
        except KeyError:
            return delayed(self["severe_cases", idx] * self.HFR, self.hospitalization_period, K)
        else:
            return delayed(critical * self.ICUFR, self.icu_period, K)

    def get_data_critical(self, idx):
        data = self["critical_cases", idx]
        return delayed_with_discharge(data, 0, self.icu_period, self.K, positive=True)

    def get_data_severe_cases(self, idx):
        K = max(self.K, 0)
        return delayed(self["cases", idx] * self.Qsv, self.severe_delay, K)

    def get_data_critical_cases(self, idx):
        K = max(self.K, 0)
        values = self["severe_cases", idx] * (self.Qcr / self.Qsv)
        delay = self.severe_delay - self.critical_delay
        return delayed(values, delay, K)
