# flake8: noqa
import datetime as dt
from datetime import date, datetime, time, timedelta

from sidekick import import_later as _imp
from sidekick import fn, X, Y, F, X_i, L, placeholder as _
import mundi
import mundi_demography as mdm
import mundi_healthcare as mhc

from .jupyter import *
from .packages import *
from .models import *
from .utils import *
from .diseases import *
from .formulas import *
from . import region

sk = _imp("sidekick")
h = _imp("hyperpython:h")
np = _imp("numpy")
pd = _imp("pandas")
plt = _imp("matplotlib.pyplot")
sns = _imp("seaborn")
sm = _imp("statsmodels.api")
smf = _imp("statsmodels.formula.api")
pydemic = _imp("pydemic")


#
# Useful constants in a Jupyter notebook
#
DAY = dt.timedelta(days=1)


def evil():
    """
    Enable sidekick's forbidden functional powers.
    """
    from sidekick.evil import forbidden_powers

    forbidden_powers()


def documentation_mode():
    """
    Monkey patch random things before executing a notebook that will be part of
    documentation.
    """
    from matplotlib.axes import SubplotBase

    SubplotBase.__repr__ = lambda _: ""
