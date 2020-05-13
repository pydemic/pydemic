from .disease import Disease
from .utils import lazy_stored_string


class Covid19(Disease):
    """
    COVID-19 first cases was detected in 2019 in China and started to spread
    worldwide early 2020. It was shortly afterwards declared a pandemic and
    a global health crisis.
    """

    name = "COVID-19"
    full_name = "SARCov-19"

    MORTALITY_TABLE_DEFAULT = "verity"
    HOSPITALIZATION_TABLE_DEFAULT = "verity"
    MORTALITY_TABLE_DESCRIPTIONS = {
        "verity": lazy_stored_string("covid-19/mortality-table-verity.txt")
    }

    def epidemic_curve(self, region, api="auto", **kwargs):
        """
        Load epidemic curve for the given region from the internet.
        """
        from .covid19_api import epidemic_curve

        return epidemic_curve(region, api, **kwargs)

    def R0(self, **kwargs):
        return 2.74

    def rho(self, **kwargs):
        return 0.45

    def prob_critical(self, **kwargs):
        return self.CFR(**kwargs) / 0.49

    def icu_period(self, **kwargs):
        return 7.5

    def hospitalization_period(self, **kwargs):
        return 7.0

    def critical_delay(self, **kwargs):
        return 1.0

    def severe_delay(self, **kwargs):
        return 5.0  # FIXME: find reference

    def hospitalization_overflow_bias(self, **kwargs):
        return 0.25

    def infectious_period(self, **kwargs):
        return 3.47

    def incubation_period(self, **kwargs):
        return 3.69
