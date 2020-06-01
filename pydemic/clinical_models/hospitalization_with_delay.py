from .crude_FR import CrudeFR
from .utils import delayed, delayed_with_discharge
from ..params import clinical
from ..utils import param_property, sliced


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
            return delayed(self["severe_cases", idx] * self.HFR, self.severe_period, K)
        else:
            return delayed(critical * self.ICUFR, self.critical_period, K)

    def get_data_critical(self, idx):
        data = self["critical_cases"]
        period = self.critical_period
        data = delayed_with_discharge(data, 0, period, self.K, positive=True)
        return sliced(data, idx)

    def get_data_severe_cases(self, idx):
        K = max(self.K, 0)
        data = delayed(self["cases"] * self.Qsv, self.severe_delay, K)
        return sliced(data, idx)

    def get_data_critical_cases(self, idx):
        K = max(self.K, 0)
        values = self["severe_cases"] * (self.Qcr / self.Qsv)
        delay = self.severe_delay - self.critical_delay
        data = delayed(values, delay, K)
        return sliced(data, idx)
