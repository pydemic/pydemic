import datetime
import itertools
from collections.abc import Sequence
from copy import copy
from typing import List, TypeVar, TYPE_CHECKING

import numpy as np
import pandas as pd

from . import column_mapping
from .report_base import Report
from .report_single import SingleReport
from .utils import RegionInfo, DefaultKeyDict
from .. import models

if TYPE_CHECKING:
    from .report_single import SingleReport

T = TypeVar("T")


class GroupReport(Sequence, Report):
    """
    A report class processes data of many simulations and expose them in easily
    consumed objects such as data frames, excel spreadsheets, PDF reports and
    more.

    Group report objects also expose a sequence interface and behave as a
    sequence of models that can be inspected and manipulated in familiar ways.
    """

    @classmethod
    def from_options(cls, factory, **kwargs) -> "GroupReport":
        if not callable(factory):
            factory = getattr(models, factory)
        expand = lambda k, xs: ((k, x) for x in xs)
        permutations = itertools.product(*(expand(k, xs) for k, xs in kwargs.items()))
        return cls([factory(**dict(args)) for args in permutations])

    _reports: List["SingleReport"]
    _region_info: DefaultKeyDict
    _column_mapping = {
        "region.sus_macro_id": column_mapping.sus_macro_id,
        "region.sus_macro_name": column_mapping.sus_macro_name,
    }

    def __init__(self, models, report_cls=SingleReport, **kwargs):
        Report.__init__(self, **kwargs)
        self._reports = [report_cls(m) for m in models]
        self._region_info = DefaultKeyDict(RegionInfo)
        self._niter = 0

    def __iter__(self):
        return (r.model for r in self._reports)

    def __len__(self):
        return len(self._reports)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            new = copy(self)
            new._reports = self._reports[idx]
            return new
        return self._reports[idx].model

    def ref(self, model):
        """
        Return a reference string for given model.
        """
        return model.region.id  # FIXME: generalize this!

    def regions(self):
        """
        Iterate over unique regions.
        """
        regions = set()
        for report in self._reports:
            region = report.model.region
            if region is None or region in regions:
                continue
            yield region

    def init_cases(self: T, data=None, regions=None, raises=True, **kwargs) -> T:
        """
        Initialize all models with statistics about cases.

        It can either receive an explicit data frame with cases/deaths
        statistics, a callable object that receives a model and return
        the desired cases. If none of these are passed, it assumes that the
        cases should be initialized from the region.
        """

        kwargs["regions"] = {} if regions is None else regions
        kwargs.setdefault("real", True)
        for i, report in enumerate(self._reports):
            print(i, report.model.region.id, report.model.region.name)
            call_safe_if(raises, report, report.init_cases, data, **kwargs)
        return self

    def init_R0(self: T, *args, raises=False, **kwargs) -> T:
        """
        Initialize R0 from cases data.
        """
        for report in self._reports:
            call_safe_if(raises, report, report.init_R0, *args, **kwargs)
        return self

    def filter_cases(self: T, min_cases=0) -> T:
        """
        Remove models for regions without the minimum amount of cases.
        """
        new = copy(self)
        new._models = []
        new._reports = []
        for m, r in zip(self._models, self._reports):
            if m["cases:last"] >= min_cases:
                new._models.append(m)
                new._reports.append(r)
        return new

    def run(self: T, time, raises=True) -> T:
        """
        Simultaneously run all models by the given time.
        """

        for i, report in enumerate(self._reports):
            try:
                report.model.run(time)
            except Exception as ex:
                print(i)
                report.log_error(str(ex), code=ex)
                if raises:
                    raise
        self._niter += time
        return self

    def report_time_rows_data(self, columns, times=None, dtype=None) -> pd.DataFrame:
        """
        Create a dataframe in which each row is a different model at a different
        time.

        Args:
            columns:
                List of output columns.
            times:
                List of times in which to evaluate columns for each model. Times
                can be given as integers (understood as offsets from the first
                simulation event)
            info:
                Column names for prefix information that is included before the
                main data.
            dtype:
                Optional output data type. This only applies to the main data
                section and is ignored in the info columns.
        """
        times = [normalize_times(m, times) for m in self]
        index = [(ref, t) for ref, ts in zip(map(self.ref, self), times) for t in ts]
        data = {"ref": [ref for ref, _ in index], "date": [t for _, t in index]}

        for col in columns:
            get_column = to_column(col)
            column_data = []

            for i, report, ts in zip(itertools.count(), self._reports, times):
                if report.is_valid:
                    values = get_column(report.model.clinical(), ts)
                    column_data.extend(values)

            data[col_name(col)] = np.array(column_data)

        # Prepare result
        out = pd.DataFrame(data)
        if dtype is not None:
            blacklist = ("ref", "date")
            dtypes = {col: dtype for col in out.columns if col not in blacklist}
            out = out.astype(dtypes)
        return out

    def report_time_columns_data(
        self, columns, times=(7, 15, 30, 60), info=(), dtype=None
    ) -> pd.DataFrame:
        """
        Create a data frame from simulations by extracting all columns in the
        given list at the selected times.

        Args:
            columns:
                List of output columns.
            times:
                List of times in which to evaluate columns for each model. Times
                can be given as integers (understood as offsets from the first
                simulation event)
            info:
                Column names for prefix information that is included before the
                main data.
            dtype:
                Optional output data type. This only applies to the main data
                section and is ignored in the info columns.
        """
        locs = [t - self._niter - 1 for t in times]
        col_names = [*map(col_name, columns)]
        columns = [*map(to_column, columns)]
        rows = []
        n_times = len(times)
        n_cols = len(columns)

        for m in self:
            m = m.clinical()
            dates = m.dates[locs]
            row = [None] * (n_times * n_cols)
            rows.append(row)

            for i, col in enumerate(columns):
                values = col(m, dates)
                row[i::n_cols] = values

        index = [m.region.id for m in self]
        col_tuples = ((x, y) for x in times for y in col_names)
        col_index = pd.MultiIndex.from_tuples(col_tuples)
        data = pd.DataFrame(rows, index=index, columns=col_index)
        if dtype:
            data = data.astype(dtype)

        columns = ["R0", "region.name", "region.population", *(info or ())]
        rows = []
        for m in self:
            obj = {
                "R0": m.R0,
                "region.name": m.region.name,
                "region.population": m.region.population,
            }
            for extra in info:
                obj[extra] = self._column_mapping[extra](m)
            row = [obj[col] for col in columns]
            rows.append(row)

        columns = [("info", col) for col in columns]
        prefix = pd.DataFrame(rows, index=index, columns=columns)
        data = pd.concat([prefix, data], axis=1)
        data.columns = pd.MultiIndex.from_tuples(data.columns)
        return data


def to_column(col):
    """
    Convert column identifier into a function with signature
    ``(model, dates) -> data`` used to retrieve data from models.
    """
    if callable(col):
        return col

    def fn(model, dates):
        return model[col + ":dates"].loc[dates]

    return fn


def col_name(col):
    """
    Return a column name from its representation.
    """

    if isinstance(col, str):
        return col
    return col.__name__


def normalize_times(model, times):
    start = model.info["event", "simulation_start"].date
    if times is None:
        return model.dates[model.dates >= start]
    elif all(isinstance(t, int) for t in times):
        return pd.to_datetime([start + datetime.timedelta(days=t) for t in times])
    else:
        raise NotImplementedError("invalid times")


def call_safe_if(*args, **kwargs):
    """
    If first argument is True, call function safely using :func:`call_safe`, else
    call function normally and propagate exceptions.
    """
    raises, *args = args
    return call_safe(*args, **kwargs)


def call_safe(*args, **kwargs):
    """
    Safely call function that involves a report. If function raises an error,
    log the error with report.log_error() method.
    """

    report, func, *args = args
    if not callable(func):
        func = getattr(report, func)
    try:
        return func(*args, **kwargs)
    except Exception as ex:
        msg = f"{type(ex).__name__}: {ex}"
        report.log_error(msg, code=ex)
        return None
