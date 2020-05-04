import os
from pathlib import Path

import pandas as pd

DATABASES = Path(__file__).absolute().parent / "databases"
EXT_PRIORITY = {
    ".pkl.bz2": pd.read_pickle,
    ".pkl.gz": pd.read_pickle,
    ".pkl": pd.read_pickle,
    ".csv": lambda p: pd.read_csv(p, index_col=0, parse_dates=True),
}
EXT_PRIORITY_WITH_KEYS = {
    ".hdf5": pd.read_hdf,
    ".sqlite": (lambda p, tb: pd.read_sql(f"SELECT * FROM {tb};", p)),
}


def read_table(ref: str, key=None):
    """
    Read data in a table file.
    """
    *dirs, name = ref.split("/")
    n = len(name)
    basedir = DATABASES
    for directory in dirs:
        basedir = basedir / directory

    paths = {f[n:]: basedir / f for f in os.listdir(basedir) if f.startswith(name)}
    if key:
        priorities = EXT_PRIORITY_WITH_KEYS
        reader = lambda f, p: f(p, key)
    else:
        priorities = EXT_PRIORITY
        reader = lambda f, p: f(p)

    for ext, fn in priorities.items():
        try:
            path = paths[ext]
        except KeyError:
            continue
        else:
            return reader(fn, path)
    raise ValueError(f"no data source found for {ref!r}")
