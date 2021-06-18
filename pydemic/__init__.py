"""
Pydemic

Epidemiological calculator tuned specifically for COVID-19.
"""
# flake8: noqa
__version__ = "0.1.11"
__author__ = "Fábio Mendes"

from typing import TYPE_CHECKING

from .params import Params, get_param, select_param

if TYPE_CHECKING:
    from . import formulas
    from . import config
    from . import db
    from . import utils
    from . import diseases
    from . import clinical_models
    from . import fitting
    from .diseases import disease
    from .model_group import ModelGroup
    from .logging import log


def __getattr__(name):
    from sidekick.api import import_later, touch

    # We want to fool Pycharm static checker to link to real modules
    models = import_later(".models", package=__package__)
    formulas = import_later(".formulas", package=__package__)
    config = import_later(".config", package=__package__)
    db = import_later(".db", package=__package__)
    utils = import_later(".utils", package=__package__)
    diseases = import_later(".diseases", package=__package__)
    clinical_models = import_later(".clinical_models", package=__package__)
    fitting = import_later(".fitting", package=__package__)
    disease = import_later(".diseases:disease", package=__package__)
    log = import_later(".logging:log", package=__package__)
    ModelGroup = import_later(".model_group:ModelGroup", package=__package__)

    try:
        value = globals()[name] = touch(locals()[name])
    except KeyError:
        raise AttributeError(name)
