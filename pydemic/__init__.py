"""
Pydemic

Epidemiological calculator tuned specifically for COVID-19.
"""
__version__ = "0.1.3"
__author__ = "Fábio Mendes"

from . import models
from . import region
from . import formulas
from . import config
from . import db
from . import utils
from . import diseases
from . import clinical_models
from . import fitting
from .diseases import disease
from .params import Params, param, get_param, select_param
from .logging import log
