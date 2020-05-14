import re
from gettext import gettext as _
from math import log10

import numpy as np

from .sequences import rpartition

__all__ = ["fmt", "pc", "pm", "p10k", "indent"]

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


def fmt(n):
    """
    Heuristically choose best format option for number.
    """

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
        return "{:.3n}".format(n * 10 ** e) + f"e-%02d" % e
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


def pc(n):
    """
    Write number as percentages.
    """
    if n == 0:
        return "0.0%"
    return fmt(100 * n) + "%"


def pm(n):
    """
    Write number as parts per thousand.
    """
    if n == 0:
        return "0.0‰"
    return fmt(1000 * n) + "‰"


def p10k(n):
    """
    Write number as parts per ten thousand.
    """
    if n == 0:
        return "0.0‱"
    return fmt(10000 * n) + "‱"


def p100k(n):
    """
    Write number as parts per 100 thousand.
    """
    if n == 0:
        return "0.0"
    return fmt(100000 * n) + "/100k"


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
    st = re.sub(r"\s+", "-", st)
    st = re.sub(r"[^\w]", "", st)
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
