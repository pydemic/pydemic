from typing import Union

from .covid19_disease import Covid19
from .disease import Disease

covid19 = Covid19("Covid-19")

DISEASE_MAP = {"covid-19": covid19}


def disease(name: Union[Disease, str]) -> Disease:
    """
    Retrieve disease by name.

    Examples:
        >>> disease('covid-19')
        Covid19()
    """
    if isinstance(name, Disease):
        return name
    try:
        return DISEASE_MAP[name.lower()]
    except KeyError:
        diseases = ", ".join(map(repr, DISEASE_MAP.keys()))
        raise ValueError(f"invalid disease. Must be one of {diseases}")
