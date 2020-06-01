from typing import Union

from .covid19_class import Covid19
from .disease_class import Disease
from .disease_params import DiseaseParams
from .utils import estimate_real_cases

covid19 = Covid19("Covid-19")
DISEASE_MAP = {"covid-19": covid19, "disease": Disease("empty")}
DEFAULT = covid19


def disease(name: Union[Disease, str] = None) -> Disease:
    """
    Retrieve disease by name.

    Examples:
        >>> disease('covid-19')
        Covid19()
    """
    if name is None:
        return DEFAULT
    if isinstance(name, Disease):
        return name
    try:
        return DISEASE_MAP[name.lower()]
    except KeyError:
        diseases = ", ".join(map(repr, DISEASE_MAP.keys()))
        msg = f"invalid disease. Must be one of {diseases}, got {name!r}"
        raise ValueError(msg)


def set_default(disease):
    """
    Set the global default disease.
    """

    global DEFAULT

    if isinstance(disease, str):
        disease = globals()["disease"](disease)
    if not isinstance(disease, Disease):
        raise TypeError(f"not a disease type: {type(disease).__name__}")
    DEFAULT = disease
