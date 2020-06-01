import re
from functools import singledispatch, lru_cache
from gettext import gettext as _
from itertools import chain
from math import log10
from numbers import Number
from types import MappingProxyType

import numpy as np

from .sequences import rpartition

N_RE = re.compile(r"(-?)(\d+)(\.\d{,2})?\d*")
identity = lambda x: x
HUMANIZED_SUFFIXES = (
    (0.1, 1, identity, "{:.04n}", ""),
    (1e2, 1, identity, "{:.03n}", ""),
    (1e3, 1, identity, "{:.2n}", ""),
    (1e6, 1, int, "{:n}", ""),
    (1e8, 1e6, identity, "{:.3n}", _("M")),
    (1e9, 1e6, identity, "{:.2n}", _("M")),
    (1e11, 1e9, identity, "{:.3n}", _("B")),
    (1e12, 1e9, identity, "{:.2n}", _("B")),
    (1e13, 1e12, identity, "{:.3n}", _("T")),
    (1e14, 1e12, identity, "{:.2n}", _("T")),
)


@singledispatch
def fmt(x, empty="-", role=None):
    """
    Format element to a humanized form.
    """
    if x is None:
        return empty
    return str(x)


@fmt.register(Number)
def __(x, empty="-", role=None):
    if role is None:
        return format_number(x, empty)
    elif role in ("pc", "%"):
        return pc(x, empty)
    elif role == "pm":
        return pm(x, empty)
    elif role == "p10k":
        return p10k(x, empty)
    elif role == "p100k":
        return p100k(x, empty)
    else:
        raise ValueError(f"invalid role: {role!r}")


def format_number(n, empty="-"):
    """
    Heuristically choose best format option for number.
    """
    if n is None:
        return empty

    if n == float("inf"):
        return _("infinity")
    try:
        return ", ".join(map(fmt, n))
    except TypeError:
        pass
    m = abs(n)

    if int(n) == n and m < 1e6:
        return "{:n}".format(int(n))

    if m < 0.01:
        e = int(-log10(m)) + 1
        return "{:.3n}".format(n * 10 ** e) + "e-%02d" % e
    for k, div, fn, fmt_, suffix in HUMANIZED_SUFFIXES:
        if m < k:
            dec = fn(m) / div
            dec -= int(dec)
            dec = fmt_.format(1 + dec)[1:]
            prefix = "{:n}".format(int(fn(n) / div))
            return prefix + dec + suffix
    return "{:.2n}".format(n)


def _fmt_aux(n, suffix=""):
    m = N_RE.match(str(n))
    sign, number, decimal = m.groups()
    return sign + _fix_int(number, 3) + (decimal or "") + suffix


def _fix_int(s, n):
    return ",".join(map("".join, rpartition(s, n)))


def pc(n, empty="-"):
    """
    Write number as percentages.
    """
    if n is None:
        return empty
    if n == 0:
        return "0.0%"
    return format_number(100 * n) + "%"


def pm(n, empty="-"):
    """
    Write number as parts per thousand.
    """
    if n is None:
        return empty
    if n == 0:
        return "0.0‰"
    return format_number(1000 * n) + "‰"


def p10k(n, empty="-"):
    """
    Write number as parts per ten thousand.
    """
    if n is None:
        return empty
    if n == 0:
        return "0.0‱"
    return format_number(10000 * n) + "‱"


def p100k(n, empty="-"):
    """
    Write number as parts per 100 thousand.
    """
    if n is None:
        return empty
    if n == 0:
        return "0.0"
    return format_number(100000 * n) + "/100k"


def indent(st, indent=4):
    """
    Indent string.
    """
    if isinstance(indent, int):
        indent = " " * indent
    return "".join(indent + ln for ln in st.splitlines(keepends=True))


def slugify(st, suffixes=(), prefixes=()):
    """
    Simplify string into a slug.

    If a list of slugified suffixes or prefixes is given, they are removed from
    the resulting string.
    """
    st = st.lower()
    st = re.sub(r"[\s_]+", "-", st)
    st = re.sub(r"[^\w-]", "", st)
    if suffixes:
        regex = "|".join(map(re.escape, suffixes))
        re.sub(fr"{regex}$", "", st)
    if prefixes:
        regex = "|".join(map(re.escape, prefixes))
        re.sub(fr"^{regex}", "", st)
    return st


def safe_int(x, default=0):
    """
    Convert float to int, with a fallback value or NaNs and Infs.
    """

    try:
        return int(x)
    except ValueError:
        if not np.isfinite(x):
            return default
        raise


def format_args(*args, **kwargs):
    """
    Return a nice string representation of the given arguments.

    Examples:
        >>> format_args(1, 2, op="sum")
        "1, 2, op='sum'"
    """
    repr_args = map(repr, args)
    repr_kwargs = (f"{k}={v!r}" for k, v in kwargs.items())
    return ", ".join(chain(repr_args, repr_kwargs))


#
# Display name functions
#
def file_type_display_name(ext, mapping=None):
    """
    Humanized representation of a file name extension.
    """
    if mapping is None:
        mapping = _file_type_display_name_mapping()

    return str(mapping.get(ext, ext.upper()))


@lru_cache(1)
def _file_type_display_name_mapping():
    return MappingProxyType(
        {
            "csv": _("CSV"),
            "csv.gz": _("zipped CSV"),
            "pkl": _("Python pickle"),
            "pkl.gz": _("Python pickle"),
            "xls": _("Excel sheet"),
            "xlsx": _("Excel sheet"),
        }
    )
