import pandas as pd


class PydemicAccessorCommon:
    """
    Common implementations for Series and DataFrame accessors.
    """

    __slots__ = ("_data",)

    @property
    def has_datetime_index(self):
        return self._data.index.dtype == "datetime"

    def __init__(self, obj):
        self._data = obj

    def reindex(self, how="datetime", origin=None, unit="D"):
        """
        Reindex data using the given method.
        """
        raise NotImplementedError


@pd.api.extensions.register_series_accessor("pydemic")
class PydemicSeriesAccessor(PydemicAccessorCommon):
    """
    Basic interface data.pydemic.<*> used to interact with empirical epidemic
    data.
    """

    __slots__ = ()

    def cases(self, *, ascertainment_rate=1.0):
        """
        Return a series with cases data, possibly correcting for ascertainment
        rate, notificaltion seasonality, and other biases.

        Args:
            ascertainment_rate:
                If given, adjust result with a given ascertaiment rate.
        """
        return self._data / ascertainment_rate


@pd.api.extensions.register_dataframe_accessor("pydemic")
class PydemicDataFrameAccessor(PydemicAccessorCommon):
    """"""

    __slots__ = ()

    def cases(self, *, ascertainment_rate=1.0):
        """
        Return a series with cases data, possibly correcting for ascertainment
        rate, notification seasonality, and other biases.

        Args:
            ascertainment_rate:
                If given, adjust result with a given ascertaiment rate.
        """
        return self._data["cases"] / ascertainment_rate
