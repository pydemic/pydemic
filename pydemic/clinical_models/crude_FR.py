from .model import ClinicalObserverModel
from .utils import delayed_with_discharge
from ..params import clinical
from ..utils import param_property, param_alias


class CrudeFR(ClinicalObserverModel):
    """
    Model in which infected can become hospitalized and suffer a constant
    hospitalization fatality rate.
    """

    params = clinical.DEFAULT

    # Primary parameters
    prob_hospitalization = param_property()
    hospitalization_period = param_property()

    # Aliases
    qh = param_alias("prob_hospitalization")

    #
    # Data methods
    #

    # Cumulative series
    def get_data_deaths(self):
        return self["cases"] * self.CFR

    def get_data_severe(self):
        data = self["severe_cases"]
        return delayed_with_discharge(data, 0, self.hospitalization_period)

    def get_data_severe_cases(self):
        return self["cases"] * self.qh
