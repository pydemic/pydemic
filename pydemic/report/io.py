import os
from typing import overload, Literal

import pandas as pd
from pandas.io.formats.style import Styler

EXTENSION_MAP = {"csv": "csv", "xls": "excel", "xlsx": "excel", "pkl": "pickle"}
COMPRESSED = {".gz", ".bz2", ".7z", ".zip"}


def data_method(func, **kwargs):
    """
    Method that dispatches to a DataFrame method.
    """
    name = func.__name__

    def method(df, **options):
        options = {**kwargs, **options}
        method = getattr(df, name)
        return method(**options)

    return staticmethod(method)


class DataFrameReader:
    """
    Reads data frames from files.

    Abstracts away several pandas functions like read_csv(), read_excel(), etc
    and dispatch according to extension type.
    """

    # Methods directly extracted from pandas
    read_csv = staticmethod(pd.read_csv)
    read_excel = staticmethod(pd.read_excel)
    read_pickle = staticmethod(pd.read_pickle)

    def read(self, path, **kwargs):
        """
        Read data from path using the most appropriate method.
        """
        method = self.__dispatch(str(path))
        return method(path, **kwargs)

    def __dispatch(self, path):
        """
        Return reader method from extension.
        """
        _, ext = os.path.splitext(str(path))
        try:
            return getattr(self, f"read_{EXTENSION_MAP[ext]}")
        except KeyError:
            if ext in COMPRESSED:
                return self.__dispatch(str(path)[: -len(ext)])


class DataFrameWriter:
    """
    Object that saves data frame objects, possibly applying styles.
    """

    save_csv = data_method(pd.DataFrame.to_csv)
    save_excel = data_method(pd.DataFrame.to_excel)
    save_pickle = data_method(pd.DataFrame.to_pickle)

    def save(self, data: pd.DataFrame, path: str, **kwargs):
        """
        Saves data to the given path.
        """
        raise NotImplementedError

    #
    # Data preparation and IO
    #
    @overload
    def style_dataframe(self, data: pd.DataFrame) -> Styler:
        ...

    @overload
    def style_dataframe(self, data: pd.DataFrame, apply: Literal[True]) -> pd.DataFrame:
        ...

    def style_dataframe(self, data, apply=False):
        """
        Style dataframe for pretty printing.
        """

        styles = {}
        dtypes = data.dtypes.to_dict()
        for col in data:
            try:
                styles[col] = self.COLUMN_STYLES[col]
            except KeyError:
                pass

            for kind, fmt in self.DTYLES_STYLES.items():
                if dtypes[col] == kind:
                    styles[col] = fmt
                    break

        if apply:
            data = data.copy()
            for k, fn in styles.items():
                fn = to_callable_formatter(fn)
                data[k] = data[k].apply(fn)
            return data

        return data.style.format(styles)


class DataFrameIO(DataFrameReader, DataFrameWriter):
    """
    Read and persist dataframes
    """


def to_callable_formatter(obj):
    if callable(obj):
        return obj
    else:
        return obj.format
