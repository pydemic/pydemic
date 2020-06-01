from abc import ABC
from collections import defaultdict, namedtuple
from functools import lru_cache
from gettext import gettext as _
from typing import TYPE_CHECKING

import pandas as pd

from pydemic.utils import fmt, pc
from .results import Results

if TYPE_CHECKING:
    from .meta_info import Meta
    from ..models import Model

SummaryRow = namedtuple("Row", ["name", "variable", "value", "default", "unit"])


class WithResultsMixin(ABC):
    """
    Subclasses that adopt this mixin gain a "summary" attribute that expose
    useful information about its owner.

    In the context of simulation models, it shows useful information about the
    evolution of the epidemic.
    """

    meta: "Meta"
    RESULT_DATES_KEYS = ("start", "end", "peak")

    @property
    def results(self) -> Results:
        return Results(self)

    def __init__(self):
        self._results_cache = defaultdict(dict)
        self._results_dirty_check = None

    def __getitem__(self, item):
        raise NotImplementedError

    #
    # Reporting and summarization
    #
    def summary_table(self: "Model", role="all", humanize=False, translate=_):
        """
        Return the summary table with model parameters.
        """

        if role == "mortality":
            params = self.disease_params
            tables = []

            try:
                tables.append(getattr(params, "mortality_table"))
            except AttributeError:
                pass

            try:
                tables.append(getattr(params, "hospitalization_table"))
            except AttributeError:
                pass

            df = pd.concat(tables, axis=1)
            df.columns = map(_, df.columns)
            df = df.applymap(pc)
            df.index = [*(f"{n}-{n + 9}" for n in df.index[:-1]), f"{df.index[-1]}+"]
            if humanize:
                rename = humanized_summary_cols(translate)
                df = df.rename(rename, axis=1)
                df.index.name = None
            return df

        if isinstance(role, str):
            role = summary_table_roles(translate)[role]

        default_params = self.disease.params()
        data = []
        for k, (name, unit, fn) in role.items():
            value = fn(getattr(self, k, None))
            default = fn(getattr(default_params, k, None))
            row = SummaryRow(name, k, value, default, unit)
            data.append(row)

        df = pd.DataFrame(data).set_index("variable")
        if humanize:
            rename = humanized_summary_cols(translate)
            df = df.rename(rename, axis=1)
            df.index.name = rename[df.index.name]

        return df

    #
    # Results methods
    #
    def get_results_keys_data(self):
        """
        Yield keys for the result["data"] dict.
        """
        yield from self.meta.variables
        yield "cases"
        yield "attack_rate"

    def get_results_value_data(self, key):
        """
        Return value for model.result["data.<key>"] queries.
        """
        if key == "attack_rate":
            population = self["population:initial"]
            return (population - self["susceptible:final"]) / population
        return self[f"{key}:final"]

    def get_results_keys_params(self):
        """
        Yield keys for the result["params"] dict.
        """
        yield from self.meta.params.primary

    def get_results_value_params(self: "Model", key):
        """
        Return value for model.result["params.<key>"] queries.
        """
        return self.get_param(key)

    def get_results_keys_dates(self):
        """
        Yield keys for the result["dates"] dict.
        """
        yield from self.RESULT_DATES_KEYS

    def get_results_value_dates(self: "Model", key):
        """
        Return value for model.result["dates.<key>"] queries.
        """
        if key == "start":
            return self.to_date(0)
        elif key == "end":
            return self.date
        elif key == "peak":
            return self["infectious:peak-date"]
        else:
            raise KeyError(key)


@lru_cache(8)
def humanized_summary_cols(translate=_):
    _ = translate
    return {
        "name": _("Name"),
        "value": _("Model value"),
        "default": _("Default value"),
        "unit": _("Unit"),
        "variable": _("Variable"),
        "severe": _("Severe"),
        "critical": _("Critical"),
    }


@lru_cache(8)
def summary_table_roles(translate=_):
    _ = translate
    epidemiological_params = {
        "R0": (_("R0"), "-", fmt),
        "duplication_time": (_("Duplication time"), _("days"), fmt),
        "infectious_period": (_("Infectious period"), _("days"), fmt),
        "incubation_period": (_("Incubation period"), _("days"), fmt),
        "prodromal_period": (_("Prodromal period"), _("days"), fmt),
        "rho": (_("Relative infectiousness"), "-", pc),
        "prob_symptoms": (_("Probability of developing symptoms"), "-", pc),
    }

    clinical_periods_params = {
        "severe_period": (_("Average hospitalization duration"), _("days"), fmt),
        "critical_period": (_("Average ICU duration"), _("days"), fmt),
        "symptoms_delay": (_("Infection to symptoms onset"), _("days"), fmt),
        "severe_delay": (_("Symptoms onset to severe"), _("days"), fmt),
        "critical_delay": (_("Symptoms onset to critical"), _("days"), fmt),
        "death_delay": (_("Symptoms onset to death"), _("days"), fmt),
    }

    clinical_rate_params = {
        "prob_severe": (_("Probability of developing severe symptoms"), "-", pc),
        "prob_critical": (_("Probability of developing critical symptoms"), "-", pc),
        "CFR": (_("Case fatality ratio"), "-", pc),
        "IFR": (_("Infection fatality ratio"), "-", pc),
        "HFR": (_("Hospitalization fatality ratio"), "-", pc),
        "ICUFR": (_("ICU fatality ratio"), "-", pc),
    }

    return {
        "epidemic": epidemiological_params,
        "clinical_periods": clinical_periods_params,
        "clinical_rates": clinical_rate_params,
        "clinical": {**clinical_rate_params, **clinical_periods_params},
        "all": {**epidemiological_params, **clinical_periods_params, **clinical_rate_params},
    }
