import numpy as np
import pandas as pd

from .crude_FR import CrudeFR
from ..params import clinical
from ..utils import param_property, param_alias


class HospitalizationWithDelay(CrudeFR):
    """
    Model in which infected can become hospitalized and suffer a constant
    hospitalization fatality rate.
    """

    params = clinical.DEFAULT

    # Primary parameters
    onset_to_hospitalization = param_property()
    hospital_fatality_rate = param_property()

    # Aliases
    qh = param_alias("prob_hospitalization")
    HFR = param_alias("hospital_fatality_rate")

    # Columns
    def get_data_deaths(self):
        return self["hospitalizations"] * self.HFR

    def get_data_hospitalizations(self):
        hospitalized = self["cases"] * self.qh
        ts = self.times + self.onset_to_hospitalization
        data = np.interp(self.times, ts, hospitalized)
        return pd.Series(data, index=self.times)
