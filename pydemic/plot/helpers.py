from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from sidekick import X

if TYPE_CHECKING:
    X: list
    Y: list
    V: list
    W: list


#
# Color handling
#
def color(i=1, ax=None):
    """
    Return the color of last i-th line in the current (or given) axis.

    This function is useful to repeat color of past plots.

    Examples:
        >>> plt.plot(Y)
        >>> plt.plot(W)
    """
    ax = ax or plt.gca()
    try:
        return ax.lines[-i].get_color()
    except IndexError:
        return None


#
# Lines
#
def mark_x(x, *args, text=None, ax=None, **kwargs):
    """
    Create a vertical line in the x position.

    Accepts all standard arguments for the plt.plot() function.

    Args:
        x (float):
            x position of the vertical line.
        text:
            Optional label written alongside the line.
        ax:
            Optional axes object.

    See Also:
        :func:`mark_y`
    """
    ax = ax or plt.gca()
    y0, y1 = ax.get_ylim()
    ax.plot([x, x], [y0, y1], *args, **kwargs)
    ax.set_ylim(y0, y1)


def mark_y(y, *args, text=None, ax=None, **kwargs):
    """
    Create a vertical line in the x position.

    Accepts all standard arguments for the plt.plot() function.

    Args:
       y (float):
           y position of the vertical line.
       text:
           Optional label written alongside the line.
       ax:
           Optional axes object.

    See Also:
        :func:`mark_x`
    """
    ax = ax or plt.gca()
    x0, x1 = ax.get_xlim()
    ax.plot([x0, x1], [y, y], *args, **kwargs)
    ax.set_xlim(x0, x1)


#
# Layout
#
def tight(which="both", ax=None):
    """
    Remove margins in either 'x', 'y' or 'both' directions.

    Examples:
        >>> plt.plot(Y)
        >>> tight('x')
    """
    if which == "x":
        x, y = True, False
    elif which == "y":
        x, y = False, True
    elif which == "both":
        x = y = True
    elif which == "none":
        return
    else:
        raise ValueError(f"invalid axis: {which!r}")

    ax = ax or plt.gca()
    lines = ax.lines

    if x:
        lines = [line.get_xdata()[[0, -1]] for line in lines]
        vmin = min(map((X[0]), lines))
        vmax = max(map((X[1]), lines))
        ax.set_xlim(vmin, vmax)
    if y:
        lines = [line.get_ydata()[[0, -1]] for line in lines]
        vmin = min(map((X[0]), lines))
        vmax = max(map((X[1]), lines))
        ax.set_ylim(vmin, vmax)


#
# Dates
#
# ... TODO
