from abc import ABC
from collections import namedtuple
from datetime import date
from functools import lru_cache
from gettext import gettext as _
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping

import pandas as pd
import sidekick.api as sk

from ..types import BoundComputedDict, ComputedDict, transform
from ..utils import fmt, pc

if TYPE_CHECKING:
    from .meta_info import Meta

SummaryRow = namedtuple("Row", ["name", "variable", "value", "default", "unit"])
mk_field = lambda name: transform(lambda model: model[f"{name}:final"])


class ResultData(ComputedDict):
    """
    Aliases for model["key:final"] values.
    """

    cases: float = mk_field("cases")
    deaths: float = mk_field("deaths")
    critical: float = mk_field("critical")
    severe: float = mk_field("severe")

    @transform
    def attack_rate(model):
        N = model["population:initial"]
        S = model["susceptible:final"]
        return (N - S) / N


class ResultDates(ComputedDict):
    """
    Important dates in model.results["dates"] dictionary.
    """

    start: date = transform(lambda model: model.to_date(0))
    end: date = transform(lambda model: model.date)
    peak: date = transform(lambda model: model["infectious:peak-date"])


class ModelDict(BoundComputedDict):
    model = sk.alias("object")


class Results(ModelDict):
    """
    Results objects store dynamic information about a simulation.

    Results data is not necessarily static throughout the simulation. It can
    include values like total death toll, number of infected people, etc. It also
    distinguishes from information stored in the model mapping interface in
    that it does not include time series.

    Most information available as ``m[<param>]`` will also be available as
    ``m.results[<param>]``. While the first typically include the whole time
    series for the object, the second typically correspond to the last value
    in the time series.
    """

    params: Mapping = transform(lambda model: MappingProxyType(model.params))
    dates: Mapping = transform(lambda model: ResultDates().partial(model))
    data: Mapping = transform(lambda model: ResultData().partial(model))

    def summary_table(self, role="all", humanize=False, translate=_):
        """
        Return the summary table with model parameters.
        """
        model = self.model

        if role == "mortality":
            params = model.disease_params
            tables = []

            for col in ["mortality_table", "hospitalization_table"]:
                try:
                    tables.append(getattr(params, col))
                except AttributeError:
                    pass
            if not tables:
                return pd.DataFrame([[]])

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

        default_params = model.disease.params()
        data = []
        for k, (name, unit, fn) in role.items():
            value = fn(getattr(model, k, None))
            default = fn(getattr(default_params, k, None))
            row = SummaryRow(name, k, value, default, unit)
            data.append(row)

        df = pd.DataFrame(data).set_index("variable")
        if humanize:
            rename = humanized_summary_cols(translate)
            df = df.rename(rename, axis=1)
            df.index.name = rename[df.index.name]

        return df

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            if key.startswith("data."):
                data = self["data"]
                return data[key[5:]]
            if key.startswith("dates."):
                data = self["dates"]
                return data[key[6:]]
            if key.startswith("params."):
                data = self["params"]
                return data[key[7:]]
            raise


class WithResultsMixin(ABC):
    """
    Subclasses that adopt this mixin gain a "summary" attribute that expose
    useful information about its owner.

    In the context of simulation models, it shows useful information about the
    evolution of the epidemic.
    """

    meta: "Meta"

    def __init__(self):
        self.results = Results(self)

    def __getitem__(self, item):
        raise NotImplementedError


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
