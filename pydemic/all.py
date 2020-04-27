import datetime as dt
from datetime import date, datetime, time, timedelta

from sidekick import import_later as _imp
from sidekick import fn, X, Y, F, X_i, L, placeholder as _

from .models import *
from .utils import *

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