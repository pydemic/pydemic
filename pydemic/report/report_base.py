from abc import ABC
from copy import copy
from typing import TypeVar, Dict

import pandas as pd

import sidekick.api as sk
from .io import DataFrameIO
from .logger import HasLoggerMixin

NOT_GIVEN = object()
T = TypeVar("T")


class Report(HasLoggerMixin, ABC):
    """
    Base class for all reports.

    A report aggregate data and instantiate one or more models and is able to
    initialize, run, and analyze those models in a consistent manner. Reports
    save those intermediary results in cache and is able to save them in disk
    in several ways.

    Report uses a fluid API and most methods return the report itself, so it can
    be chained indefinitely.

    >>> (
    ...     report
    ...         .init_data()
    ...         .init_R0()
    ...         .run(60)
    ...         .report_columns(['cases', 'deaths'])
    ...         .save('data.xls')
    ...         .report_params()
    ...         .save('params.xls')
    ... )
    """

    # Class level constants
    COLUMN_STYLES = {"time": int}
    DTYPE_STYLES = {"float": lambda x: round(x, 2)}

    # Attributes
    results: Dict[str, pd.DataFrame]
    is_valid: bool
    io: DataFrameIO = sk.lazy(lambda _: DataFrameIO())

    def __init__(self, io=None, logger=None):
        HasLoggerMixin.__init__(self, logger)
        self.results = {}
        self._last_result = None
        if io is not None:
            self.io = io

    def copy(self: T) -> T:
        """
        Return a copy of report instance, clearing analysis and results.
        """
        new = copy(self)
        new.results = dict(self.results)
        return new

    def store_result(self: T, data, name: str = None) -> T:
        """
        Save result in the given section of the results dictionary.
        """

        self.results[name] = data
        self._last_result = name
        return self

    def save(self: T, path, name=NOT_GIVEN, **kwargs) -> T:
        """
        Save result as a file in the given path. The method of persistence is
        chosen from extension.
        """

        try:
            data = self.results[self._last_result if name is NOT_GIVEN else name]
        except KeyError:
            raise RuntimeError(f"no result for {name} was computed.")
        self.io.save(data, path, **kwargs)
        return self
