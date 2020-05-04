from .disease import Disease, lazy_stored_string


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
