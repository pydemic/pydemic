import re
import warnings
from abc import ABC
from pathlib import Path

import numpy as np
import pandas as pd
import sidekick as sk
from sidekick import X

import mundi
from .disease_params import DiseaseParams
from .utils import (
    QualDataT,
    read_table,
    Dataset,
    QualValueT,
    world_age_distribution,
    age_adjusted_average,
    set_age_distribution_default,
    estimate_real_cases,
)
from .. import db
from .. import fitting as fit
from ..config import set_cache_options
from ..types import ImproperlyConfigured
from ..utils import to_json

# Types
set_cache_options("diseases-api", compress=True)


class Disease(ABC):
    """
    Basic interface that exposes information about specific diseases.
    """

    # Attributes
    name: str = ""
    description: str = ""
    full_name: str = ""
    path: str = sk.lazy(lambda _: "diseases/" + _.name.lower().replace(" ", "-"))
    abspath: Path = sk.lazy(lambda _: db.DATABASES / _.path)
    _default_params = sk.lazy(lambda _: DiseaseParams(_))

    # Constants
    MORTALITY_TABLE_DEFAULT = "default"
    MORTALITY_TABLE_ALIASES = {}
    MORTALITY_TABLE_DESCRIPTIONS = {}
    HOSPITALIZATION_TABLE_DEFAULT = "default"
    HOSPITALIZATION_TABLE_ALIASES = {}
    HOSPITALIZATION_TABLE_DESCRIPTIONS = {}
    PARAMS_BLACKLIST = frozenset({"to_json", "to_record", "to_dict", "params", "epidemic_curve"})

    def __init__(self, name=None, description=None, full_name=None, path=None):
        self.name = name or self.name or type(self).__name__
        self.full_name = full_name or self.full_name
        self.description = description or self.description or self.__doc__.strip()
        self.path = path or self.path

    def __str__(self):
        return self.full_name

    def __repr__(self):
        return f"{type(self).__name__}()"

    def __bool__(self):
        return True

    def mortality_table(
        self, source: str = None, qualified=False, extra=False, region=None
    ) -> QualDataT:
        """
        Return the mortality table of the disease.

        The mortality table is stratified by age and has at least two
        columns, one with the CFR and other with the IFR per age group.

        Args:
            source (str):
                Select source/reference that collected this data. Different
                datasets may be available representing the best knowledge by
                different teams or assumptions about the disease. This argument
                is a string identifier for that version of the data.
            qualified (bool):
                If True, return a :cls:`Dataset` namedtuple with
                (data, source, notes) attributes.
            extra (bool):
                If True, display additional columns alongside with ["IRF", "CFR"].
                The extra columns can be different for each dataset.
        """
        df = self._read_dataset("mortality-table", source, qualified)
        return df if extra else df[["IFR", "CFR"]]

    def hospitalization_table(
        self, source=None, qualified=False, extra=False, region=None
    ) -> QualDataT:
        """
        Return the hospitalization table of the disease. The hospitalization
        table is stratified by age and has two columns: "severe" and "critical".
        Each column represents the probability of developing either "severe"
        (requires clinical treatment) or "critical" (requires ICU) cases.

        In both cases, the probabilities are interpreted as the ratio between
        hospitalization and cases, *excluding* asymptomatic individuals

        Args:
            source (str):
                Select source/reference that collected this data. Different
                datasets may be available representing the best knowledge by
                different teams or assumptions about the disease. This argument
                is a string identifier for that version of the data.
            qualified (bool):
                If True, return a :cls:`Dataset` namedtuple with
                (data, source, notes) attributes.
            extra (bool):
                If True, display additional columns alongside with ["severe",
                "critical"]. The extra columns can be different for each dataset.
        """
        df = self._read_dataset("hospitalization-table", source, qualified)
        return df if extra else df[["severe", "critical"]]

    def _read_dataset(self, which, source, qualified) -> QualDataT:
        """
        Worker method for many data reader methods.
        """
        attr = which.replace("-", "_").upper()
        aliases = getattr(self, attr + "_ALIASES")
        descriptions = getattr(self, attr + "_DESCRIPTIONS")
        source = source or getattr(self, attr + "_DEFAULT")

        if ":" in source:
            name, _, arg = source.partition(":")
            name = aliases.get(name, name)
            method = re.sub(r"[\s-]", "_", f"get_data_{which}_{name}")
            try:
                fn = getattr(self, method)
            except AttributeError:
                raise ValueError(f"invalid method source: {source!r}")
            else:
                data = fn(arg)
                description = fn.__doc__
        else:
            name = aliases.get(source, source)
            description = descriptions.get(name, "").strip()
            data = read_table(f"{self.path}/{which}-{name}").copy()

        if qualified:
            return Dataset(data, source, description)
        return data

    def case_fatality_ratio(self, **kwargs) -> QualValueT:
        """
        Compute the case fatality ratio with possible age distribution
        adjustments.

        Args:
            age_distribution:
                A age distribution or a string with a name of a valid Mundi
                region with known demography. If not given, uses the average
                world age distribution.
            source:
                Reference source used to provide the mortality table.
        """
        return self._fatality_ratio("CFR", **kwargs)

    def infection_fatality_ratio(self, **kwargs) -> QualValueT:
        """
        Compute the infection fatality ratio with possible age distribution
        adjustments.

        Args:
            age_distribution:
                A age distribution or a string with a name of a valid Mundi
                region with known demography. If not given, uses the average
                world age distribution.
            source:
                Reference source used to provide the mortality table.
        """
        return self._fatality_ratio("IFR", **kwargs)

    def _fatality_ratio(self, col, age_distribution=None, source=None, region=None):
        table = self.mortality_table(source=source)

        if age_distribution is None and region:
            ages = mundi.region(region).age_distribution
        elif age_distribution is None:
            ages = world_age_distribution()
        else:
            ages = age_distribution
        return age_adjusted_average(ages, table[col])

    def epidemic_curve(
        self, region, diff=False, smooth=False, real=False, keep_observed=False, window=14, **kwargs
    ) -> pd.DataFrame:
        """
        Load epidemic curve for the given region.

        Args:
            region:
                Mundi r/egion or a string with region code.
            diff (bool):
                If diff=True, return the number of new daily cases instead of
                the accumulated epidemic curves.
            smooth (bool):
                If True, return a data frame with raw and smooth columns. The
                resulting dataframe adopts a MultiIndex created from the
                product of ["observed", "smooth"] x ["cases", "deaths"]
            real (bool, str):
                If True, estimate the real number of cases by trying to correct
                for some ascertainment rate.
            keep_observed (bool):
                If True, keep the raw observed cases under the "cases_observed"
                and "deaths_observed" columns.
            window (int):
                Size of the triangular smoothing window.
        """
        data = self._epidemic_curve(mundi.region(region), **kwargs)

        if real:
            method = "CFR" if real is True else real
            params = self.params(region=region)
            real = estimate_real_cases(data, params, method)
            if keep_observed:
                rename = {"cases": "cases_observed", "deaths": "deaths_observed"}
                data = pd.concat([real, data.rename(rename, axis=1)], axis=1)
            else:
                data = real

        if diff:
            values = np.diff(data, prepend=0, axis=0)
            data = pd.DataFrame(values, index=data.index, columns=data.columns)

        if smooth:
            columns = pd.MultiIndex.from_product([["observed", "smooth"], data.columns])
            data = pd.concat([data, fit.smooth(data, window)], axis=1)
            data.columns = columns

        return data

    def _epidemic_curve(self, region, **kwargs):
        raise NotImplementedError

    #
    # Basic epidemiology
    #
    def R0(self, **kwargs) -> QualValueT:
        """
        R0 for disease.
        """
        return NotImplemented

    def rho(self, **kwargs):
        """
        Relative transmissibility of asymptomatic cases.

        The default value is 1.0. This value makes the distinction between
        symptomatic and asymptomatic as a mere notification problem
        """
        return 1.0

    #
    # Clinical progression
    #

    # Probabilities
    def prob_severe(self, **kwargs) -> QualValueT:
        """
        Probability that a case become severe enough to recommend hospitalization.

        Estimated from "severe cases / symptomatic cases".
        """
        return self._prob_from_hospitalization_table("severe", **kwargs)

    def prob_critical(self, **kwargs) -> QualValueT:
        """
        Probability that a case become severe enough to recommend intensive
        care treatment.

        Estimated from "critical cases / symptomatic cases".
        """
        try:
            return self._prob_from_hospitalization_table("critical", **kwargs)
        except KeyError:
            pass
        try:
            return self.CFR(**kwargs) / self.icu_fatality_ratio(**kwargs)
        except RecursionError:
            model = type(self).__name__
            raise ImproperlyConfigured(
                f"""
{model} class must implement either prob_critical() or icu_fatality_ratio()
methods. Otherwise, the default implementation creates a recursion between both
methods."""
            )

    def _prob_from_hospitalization_table(self, col, **kwargs):
        """
        Worker function for prob_critical and prob_severe methods.
        """
        ages = set_age_distribution_default(kwargs, drop=True)
        values = self.hospitalization_table(**kwargs)
        return age_adjusted_average(ages, values[col])

    def prob_aggravate_to_icu(self, **kwargs) -> QualValueT:
        """
        Probability that an hospitalized patient require ICU.
        """
        return self.prob_critical(**kwargs) / self.prob_severe(**kwargs)

    def prob_symptoms(self, **kwargs) -> QualDataT:
        """
        Probability that disease develop symptomatic cases.
        """
        e = 1e-50
        ages = set_age_distribution_default(kwargs, drop=True)
        table = self.mortality_table(**kwargs)
        ratios = (table["IFR"] + e) / (table["CFR"] + e)
        return age_adjusted_average(ages, ratios)

    def hospitalization_overflow_bias(self, **kwargs) -> QualValueT:
        """
        Increase in the fraction of critical patients when severe patients are
        not treated in a proper healthcare facility.

        The default implementation ignores this phenomenon and returns zero.
        """
        return 0.0

    # Durations
    def infectious_period(self, **kwargs) -> QualValueT:
        """
        Period in which cases are infectious.
        """
        return NotImplemented

    def incubation_period(self, **kwargs) -> QualValueT:
        """
        Period between infection and symptoms onset.
        """
        return NotImplemented

    def severe_period(self, **kwargs) -> QualValueT:
        """
        Duration of the "severe" state of disease.

        The default implementation assumes it is the same as the hospitalization
        period. If this method has a different implementation, this parameter
        is interpreted as the duration of the severe state in the absence of
        hospitalization.
        """
        return self.hospitalization_period(**kwargs)

    def critical_period(self, **kwargs) -> QualValueT:
        """
        Duration of the "critical" state of disease.

        The default implementation assumes it is the same as the ICU
        period. If this method has a different implementation, this parameter
        is interpreted as the duration of the severe state in the absence of
        hospitalization.
        """
        return self.icu_period(**kwargs)

    def icu_period(self, **kwargs) -> QualValueT:
        """
        Duration of ICU treatment.

        The default implementation assumes this to be zero, meaning the disease
        never progress to a critical state or kills instantly if this state
        is reached.

        Obviously, most diseases would need to override this method.
        """
        return 0.0

    def hospitalization_period(self, **kwargs) -> QualValueT:
        """
        Duration of hospitalization treatment.

        The default implementation assumes this to be zero, meaning the disease
        never progress to a severe state or kills instantly if this state
        is reached.

        Obviously, most diseases would need to override this method.
        """
        return 0.0

    # Delays
    def symptom_delay(self, **kwargs):
        """
        Average duration between becoming infectious and symptom delay.
        """
        return 0.0

    def critical_delay(self, **kwargs) -> QualValueT:
        """
        Average duration between symptom onset and necessity of ICU admission.
        """
        return NotImplemented

    def severe_delay(self, **kwargs) -> QualValueT:
        """
        Average duration between symptom onset and necessity of hospital admission.
        """
        return NotImplemented

    def death_delay(self, **kwargs):
        """
        Average duration between symptom onset and necessity of hospital
        admission.
        """
        return self.critical_delay(**kwargs) + self.icu_period(**kwargs)

    # Derived values
    def gamma(self, **kwargs):
        """
        The inverse of infectious period.
        """
        return 1 / self.infectious_period(**kwargs)

    def sigma(self, **kwargs):
        """
        The inverse of incubation period.
        """
        return 1 / self.incubation_period(**kwargs)

    def hospital_fatality_ratio(self, **kwargs) -> QualValueT:
        """
        Probability of death once requires hospitalization.
        """
        # FIXME: make properly age-stratified
        return self.CFR(**kwargs) / self.prob_severe(**kwargs)

    def icu_fatality_ratio(self, **kwargs) -> QualValueT:
        """
        Probability of death once requires intensive care.

        The default implementation assumes that death occurs only after reaching
        a critical state.
        """
        # FIXME: make properly age-stratified
        return self.CFR(**kwargs) / self.prob_critical(**kwargs)

    # Aliases
    def Qs(self, **kwargs) -> QualValueT:
        """
        Alias to "prob_symptoms".
        """
        return self.prob_symptoms(**kwargs)

    def Qsv(self, **kwargs) -> QualValueT:
        """
        Alias to "prob_severe".
        """
        return self.prob_severe(**kwargs)

    def Qcr(self, **kwargs) -> QualValueT:
        """
        Alias to "prob_critical".
        """
        return self.prob_critical(**kwargs)

    def CFR(self, **kwargs) -> QualValueT:
        """
        Alias to "case_fatality_ratio"
        """
        return self.case_fatality_ratio(**kwargs)

    def IFR(self, **kwargs) -> QualValueT:
        """
        Alias to "infection_fatality_ratio"
        """
        return self.infection_fatality_ratio(**kwargs)

    def HFR(self, **kwargs) -> QualValueT:
        """
        Alias to "hospital_fatality_ratio"
        """
        return self.hospital_fatality_ratio(**kwargs)

    def ICUFR(self, **kwargs) -> QualValueT:
        """
        Alias to "icu_fatality_ratio"
        """
        return self.icu_fatality_ratio(**kwargs)

    #
    # Conversions to parameters
    #
    def params(self, *args, **kwargs) -> DiseaseParams:
        """
        Wraps disease in a parameter namespace.

        This is useful and a safer alternative to use disease as an argument
        to several functions that expect params.
        """
        if not args and not kwargs:
            return self._default_params
        return DiseaseParams(self, *args, **kwargs)

    def to_record(self, **kwargs) -> sk.record:
        """
        Return a sidekick record object with all disease parameters.
        """
        return sk.record(self.to_dict(**kwargs))

    def to_dict(self, *, alias=False, transform=False, **kwargs) -> dict:
        """
        Return a dict with all epidemiological parameters.
        """

        methods = (
            "R0",
            "rho",
            "case_fatality_ratio",
            "infection_fatality_ratio",
            "hospital_fatality_ratio",
            "icu_fatality_ratio",
            "infectious_period",
            "incubation_period",
            "hospitalization_period",
            "icu_period",
            "hospitalization_overflow_bias",
            "severe_period",
            "critical_period",
            "prob_symptoms",
            "prob_severe",
            "prob_critical",
        )

        aliases = (
            ("Qsv", "prob_severe"),
            ("Qcr", "prob_critical"),
            ("Qs", "prob_symptoms"),
            ("IFR", "infection_fatality_ratio"),
            ("CFR", "case_fatality_ratio"),
            ("HFR", "hospital_fatality_ratio"),
            ("ICUFR", "icu_fatality_ratio"),
        )

        transforms = (
            ("gamma", "infectious_period", (1 / X)),
            ("sigma", "incubation_period", (1 / X)),
        )

        out = {}
        for method in methods:
            try:
                out[method] = getattr(self, method)(**kwargs)
            except Exception as e:
                name = type(e).__name__
                warnings.warn(f"raised when calling disease.{method}():\n" f'    {name}: "{e}" ')
                raise

        if alias:
            for k, v in aliases:
                if v in out:
                    out[k] = out[v]

        if transform:
            for k, v, fn in transforms:
                if v in out:
                    out[k] = fn(out[v])

        return out

    def to_json(self, **kwargs) -> dict:
        """
        Similar to :meth:`to_dict`, but converts non-compatible values such as
        series and dataframes to json.
        """
        return to_json(self.to_dict(**kwargs))
