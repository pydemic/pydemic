import numpy as np
import pandas as pd

from .model import ClinicalObserverModel
from ..params import clinical
from ..utils import param_property, param_alias


class DelayedHospitalization(ClinicalObserverModel):
    """
    Model in which infected can become hospitalized and suffer a constant
    hospitalization fatality rate.
    """

    params = clinical.DEFAULT

    # Primary parameters
    onset_to_hospitalization = param_property()
    hospitalization_period = param_property()
    prob_hospitalization = param_property()
    hospital_fatality_rate = param_property()

    # Aliases
    qh = param_alias("prob_hospitalization")
    HFR = param_alias("hospital_fatality_rate")

    # Columns
    def get_data_hospitalizations(self):
        hospitalized = self.cases * self.qh
        ts = self.times + self.onset_to_hospitalization
        data = np.interp(self.times, ts, hospitalized)
        return pd.Series(data, index=self.dates)

    def get_data_hospitalized(self):
        qh = self.qh
        cases = self.cases
        te = self.times + self.onset_to_hospitalization
        td = self.times + self.hospitalization_period
        entry = np.interp(self.times, te, cases) * qh
        discharge = np.interp(self.times, td, entry)

        # Currently hospitalized
        data = np.maximum(entry - discharge, 0.0)
        return pd.Series(data, index=self.dates)

    def get_data_fatalities(self):
        return self["hospitalizations"] * self.HFR