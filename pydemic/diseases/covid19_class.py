from .disease_class import Disease
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

    def _epidemic_curve(self, region, api="auto", **kwargs):
        """
        Load epidemic curve for the given region from the internet.
        """
        from ..api.covid19 import epidemic_curve

        return epidemic_curve(region, api, **kwargs)

    def R0(self, **kwargs):
        return 2.74

    def rho(self, **kwargs):
        return 0.45

    def icu_period(self, **kwargs):
        return 7.5

    def hospitalization_period(self, **kwargs):
        return 10.0

    def critical_delay(self, **kwargs):
        # FIXME: find reference.
        # We estimated this parameter from anecdotal evidence from physicians
        # that work in hospitals and non-official compilations of cases in
        # Brazil.
        #
        # This parameter should be available in the literature, however.
        return 1.0

    def severe_delay(self, **kwargs):
        return 3.3

    def hospitalization_overflow_bias(self, **kwargs):
        # We do not have this parameter and it seems very unlikely that it will
        # ever be measured directly. This correspond to the fraction of
        # hospitalized cases that evolve to critical state in the absence of
        # treatment *in excess* of the natural evolution of the disease.
        return 0.25

    def infectious_period(self, **kwargs):
        return 3.47

    def incubation_period(self, **kwargs):
        return 3.69
