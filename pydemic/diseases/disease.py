import re
from abc import ABC
from functools import lru_cache
from numbers import Number
from pathlib import Path
from typing import NamedTuple, Union

import numpy as np
import pandas as pd
import sidekick as sk

import mundi
from mundi_demography import age_distribution
from .. import db
from ..config import set_cache_options
from ..utils import slugify

DB_PATH = db.DATABASES / "diseases"
not_implemented = property(lambda _: sk.error(NotImplementedError))
set_cache_options("diseases-api", compress=True)


class Dataset(NamedTuple):
    """
    A qualified dataset. Includes raw data in the .data attribute and meta
    information
    """

    data: Union[pd.DataFrame, pd.Series, np.ndarray, Number]
    source: str = ""
    notes: str = ""


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

    # Constants
    MORTALITY_TABLE_DEFAULT = not_implemented
    MORTALITY_TABLE_ALIASES = {}
    MORTALITY_TABLE_DESCRIPTIONS = {}
    HOSPITALIZATION_TABLE_DEFAULT = not_implemented
    HOSPITALIZATION_TABLE_ALIASES = {}
    HOSPITALIZATION_TABLE_DESCRIPTIONS = {}

    def __init__(self, name=None, description=None, full_name=None, path=None):
        self.name = name or self.name or type(self.__name__)
        self.full_name = full_name or self.full_name
        self.description = description or self.description or self.__doc__.strip()
        self.path = path or self.path

    def __str__(self):
        return self.full_name

    def __repr__(self):
        return f"{type(self).__name__}()"

    def mortality_table(self, source: str = None, qualified=False, extra=False):
        """
        Return the mortality table of the disease.

        The mortality table is stratified by age and has at least two
        columns, one with the CFR and other with the IFR.

        Args:
            source (str):
                Allow one to choose the source from that data if different
                versions were collected by different teams or in different
                scenarios. This argument is a string identifier for that version
                of the data.
            qualified (bool):
                If True, return a :cls:`Dataset` namedtuple with
                (data, source, notes) attributes.
            extra (bool):
                If True, display additional columns alongside with ["IRF", "CFR"].
                The extra columns are variable for each dataset.
        """
        df = self._read_dataset("mortality-table", source, qualified)
        return df if extra else df[["IFR", "CFR"]]

    def hospitalization_table(self, source=None, qualified=False, extra=False):
        """
        Return the hospitalization table of the disease.

        This table is stratified by age and can have many columns
        depending on the clinical progression of the disease. Typically, it will
        feature a "hospitalization" column. For some diseases, we have more
        detailed information about the disease progression including "icu"
        admissions, need for "ventilators", etc.
        """
        df = self._read_dataset("hospitalization-table", source, qualified)
        return df if extra else df[["hospitalization"]]

    def _read_dataset(self, which, source, qualified):
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

    def CFR(self, ages=None, source=None):
        """
        Compute the case fatality ratio with possible age distribution
        adjustments.

        Args:
            ages:
                A age distribution or a string with a name of a valid Mundi
                region with known demography. If not given, uses the average
                world age distribution.
            source:
                Reference source used to provide the mortality table.
        """
        return self._fatality_rate("CFR", ages, source)

    def IFR(self, ages=None, source=None):
        """
        Compute the infection fatality ratio with possible age distribution
        adjustments.

        Args:
            ages:
                A age distribution or a string with a name of a valid Mundi
                region with known demography. If not given, uses the average
                world age distribution.
            source:
                Reference source used to provide the mortality table.
        """
        return self._fatality_rate("IFR", ages, source)

    def _fatality_rate(self, col, ages, source):
        table = self.mortality_table(source=source)
        ages = world_age_distribution() if ages is None else ages
        return age_adjusted_average(ages, table[col])

    def epidemic_curve(self, region, **kwargs):
        """
        Load epidemic curve for the given region.
        """
        raise NotImplementedError


def normalize_source(source):
    """
    Slugify and remove unnecessary suffixes from source reference.

    This is useful to transform a textual description of the source work to
    a valid filename in which it is installed.

    Examples:
        >>> normalize_source("Verity, et. al.")
        "verity"
    """
    return slugify(source, suffixes=("et-al",))


def lazy_stored_string(path):
    """
    A lazy string-like object that loads information from a file.
    """
    return sk.deferred(_lazy_description, path)


def age_adjusted_average(ages, value):
    """
    Given an age distribution and a table with age-adjusted values, compute the
    mean value for population.

    This method automatically aligns ages and value if both tables are not
    aligned.
    """
    if isinstance(ages, str):
        ages = age_distribution(ages)
    population = ages.values.sum()
    if set(ages.index).issuperset(value.index):
        value = value.reindex(ages.index, method="ffill")
    else:
        raise NotImplementedError
    return (value * ages / population).sum()


@lru_cache(1)
def world_age_distribution():
    """
    World age distribution computed by summing
    """
    countries = mundi.countries()
    return countries.mundi["age_distribution"].sum(0)


@lru_cache(32)
def _lazy_description(path):
    with open(DB_PATH / path) as fd:
        return fd.read()


@lru_cache(32)
def read_table(path: str, key=None):
    return db.read_table(path, key)
