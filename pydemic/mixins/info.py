from typing import Any, Sequence

import pandas as pd

from mundi import Region


class Info:
    """
    Info objects store static information about a simulation.

    Not all information exposed by this object is necessarily relevant to a
    simulation model, but might be useful for later analysis or for
    convenience.

    Info attributes are organized in a dotted namespace.
    """

    __slots__ = ("owner",)
    owner: Any

    def __init__(self, owner):
        self.owner = owner

    def __getitem__(self, item):
        raise NotImplementedError

    def to_dict(self, which="default") -> dict:
        """
        Expose information as dictionary.
        """
        if which == "default":
            raise NotImplementedError
        elif not isinstance(which, str) and isinstance(which, Sequence):
            return {k: self[k] for k in which}
        else:
            return flatten_dict(self[which], which + ".")

    def to_series(self, which="default") -> pd.Series:
        """
        Expose information as a series object.
        """
        return pd.Sereis(flatten_dict(self.to_dict(which)))


class RegionalInfo(Info):
    """
    Info object that exposes information about regions.
    """

    __slots__ = ()
    owner: Any
    region: Region

    @property
    def region(self):
        region = getattr(self.owner, "region")
        if region is None:
            cls_name = type(self.owner).__name__
            msg = f"the {cls_name} instance was not initialized with a region!"
            raise RuntimeError(msg)
        if isinstance(region, str):
            region = Region(region)
        return region

    def __getitem__(self, item):
        try:
            return super()[item]
        except KeyError:
            return self.region[item]

    def get_info_demography(self, arg):
        """
        Retrieve demographic parameters about the population.
        """
        if arg == "population":
            return self.data.iloc[0].sum()
        elif arg == "age_distribution":
            return self.age_distribution
        elif arg == "age_pyramid":
            return self.age_pyramid
        elif arg == "seniors":
            return self.age_distribution.loc[60:].sum()
        else:
            raise ValueError(f"unknown argument: {arg!r}")

    def get_info_healthcare(self, arg):
        """
        Return info about the healthcare system.
        """
        if arg == "icu_total_capacity":
            return 4 * self.icu_capacity
        elif arg == "hospital_total_capacity":
            return 4 * self.hospital_capacity
        raise NotImplementedError
