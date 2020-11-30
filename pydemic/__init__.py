"""
Pydemic

Epidemiological calculator tuned specifically for COVID-19.
"""
# flake8: noqa
__version__ = "0.1.11"
__author__ = "Fábio Mendes"

# noinspection PyUnresolvedReferences
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
from .params import Params, get_param, select_param
from .types import param
from .model_group import ModelGroup
from .logging import log


# Fix old sidekick versions
import sidekick as _sidekick

_sidekick.partition_at = _sidekick.partition
