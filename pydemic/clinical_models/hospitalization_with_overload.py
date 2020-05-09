import numpy as np
import pandas as pd
import sidekick as sk
from scipy.integrate import cumtrapz

from .hospitalization_with_delay import HospitalizationWithDelay, healthcare


class HospitalizationWithOverload(HospitalizationWithDelay):
    """
    Hospitals have a maximum number of ICU and regular clinical beds.
    Death rates increase when this number overflows.
    """

    @property
    def region(self):
        region = getattr(self.infection_model, "region", None)
        if region is None:
            raise RuntimeError("infectious model has no defined region.")
        return region

    @sk.lazy
    def icu_capacity(self):
        return healthcare.icu_surge_capacity(self.region)

    @sk.lazy
    def hospital_capacity(self):
        return healthcare.hospital_surge_capacity(self.region)

    #
    # Data methods
    #

    # Deaths
    def get_data_deaths(self):
        return self["natural_deaths"] + self["overflow_deaths"]

    def get_data_natural_deaths(self):
        """
        The number of deaths assuming healthcare system in full capacity.
        """
        return super().get_data_deaths()

    def get_data_overflow_deaths(self):
        """
        The number of deaths caused by overflowing the healthcare system.
        """
        return self["icu_overflow_deaths"] + self["hospital_overflow_deaths"]

    def get_data_icu_overflow_deaths(self):
        """
        The number of deaths caused by overflowing ICUs.
        """
        area = cumtrapz(self["critical_overflow"], self.times, initial=0)
        return pd.Series(area / self.icu_period, index=self.times)

    def get_data_hospital_overflow_deaths(self):
        """
        The number of deaths caused by overflowing regular hospital beds.
        """
        area = cumtrapz(self["severe_overflow"], self.times, initial=0)
        cases = area / self.hospitalization_period
        deaths = cases * min(self.qc * self.qc_overflow_bias, 1)
        return pd.Series(deaths, index=self.times)

    def get_data_overflow_death_rate(self):
        """
        Daily number of additional deaths due to overflowing the healthcare system.
        """
        return self["overflow_deaths"].diff().fillna(0)

    def get_data_icu_overflow_death_rate(self):
        """
        Daily number of additional deaths due to overflowing the ICU capacity.
        """
        return self["icu_overflow_deaths"].diff().fillna(0)

    def get_data_hospital_overflow_death_rate(self):
        """
        Daily number of additional deaths due to overflowing hospital capacity.
        """
        return self["hospital_overflow_deaths"].diff().fillna(0)

    # Severe/hospitalizations dynamics
    def get_data_severe_overflow(self):
        """
        The number of severe cases that are not being treated in a hospital
        facility.
        """
        data = np.maximum(self["severe"] - self.hospital_capacity, 0)
        return pd.Series(data, index=self.times)

    def get_data_hospitalized_cases(self):
        area = cumtrapz(self["hospitalized"], self.times, initial=0)
        return pd.Series(area / self.hospitalization_period, index=self.times)

    def get_data_hospitalized(self):
        demand = self["severe"]
        data = np.minimum(demand, self.hospital_capacity)
        return pd.Series(data, index=self.times)

    # Critical/ICU dynamics
    def get_data_critical_overflow(self):
        """
        The number of critical cases that are not being treated in an ICU
        facility.
        """
        data = np.maximum(self["critical"] - self.icu_capacity, 0)
        return pd.Series(data, index=self.times)

    def get_data_icu_cases(self):
        area = cumtrapz(self["icu"], self.times, initial=0)
        return pd.Series(area / self.icu_period, index=self.times)

    def get_data_icu(self):
        demand = self["hospitalized"]
        data = np.minimum(demand, self.icu_capacity)
        return pd.Series(data, index=self.times)

    # Aliases
    get_data_icu_overflow = get_data_critical_overflow
    get_data_hospital_overflow = get_data_severe_overflow
