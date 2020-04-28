import numpy as np
import pandas as pd

from .model import ClinicalObserverModel
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

    # Columns
    def get_data_deaths(self):
        return self["cases"] * self.CFR

    def get_data_hospitalizations(self):
        return self["cases"] * self.qh

    def get_data_hospitalized(self):
        hospitalizations = self["hospitalizations"]
        ts = self.times + self.hospitalization_period
        entry = np.interp(self.times, ts, hospitalizations)
        discharge = np.interp(self.times, ts, entry)

        # Currently hospitalized
        data = np.maximum(entry - discharge, 0.0)
        return pd.Series(data, index=self.times)
