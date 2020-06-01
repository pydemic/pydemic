import io
import re
from typing import Dict, Union

import sidekick as sk
from sidekick import placeholder as _

SPACES = re.compile(r"^\s+")
INDENT = re.compile(r"^\s+")
SECTION = re.compile(r"^\w+( \w+)*:$")
ARG = re.compile(r"^\w+(\s+\([^)]+\))?:$")


def docstring(**kwargs):
    """
    Replace values in the decorated function docstring.
    """

    def decorator(fn):
        fn.__doc__ = dedent(fn.__doc__).format(name=fn.__name__, module=fn.__module__, **kwargs)
        return fn

    return decorator


def parse_docstring(obj: Union[str, callable]) -> Dict[str, str]:
    """
    Parse a docstring in google format.

    Args:
        obj:
            Docstring or function

    Returns:
        A parsed docstring
    """
    if not isinstance(obj, str):
        doc = obj.__doc__
        if doc is None:
            raise ValueError("object does not have docstring")
    else:
        doc = obj
    lines = doc.splitlines()
    indent = len(lines[-1])
    lines = sk.pipe(
        lines,
        sk.map(_[indent:]),
        sk.partition_by(SECTION.fullmatch),
        sk.cons(("",)),
        sk.partition(2),
        dict,
    )
    return {k[0].strip(":"): "\n".join(filter(None, v)) for k, v in lines.items()}


def parse_args(args):
    """
    Parse the "Args" section of a google docstring.
    """
    out = sk.pipe(
        args.splitlines(), sk.map(str.lstrip), sk.partition_by(ARG.fullmatch), sk.partition(2), dict
    )
    return {k[0].rstrip(":"): "\n".join(v) for k, v in out.items()}


def render_docstring(parsed: Dict[str, str], indent=""):
    """
    Renders parsed docstring returned from parse_docstring.

    Args:
        parsed:
            A dictionary returned from parse_docstring()
        indent:
            An optional prefix to include in each line of the docstring.

    Returns:
        Reconstruct docstring from parsed values.
    """
    stream = io.StringIO()
    for k, v in parsed.items():
        if k:
            stream.write(k)
            stream.write(":\n")
        stream.write(v)
        stream.write("\n\n")

    out = stream.getvalue()
    if indent:
        out = "\n".join(indent + ln for ln in out.splitlines())
    return out


def dedent(doc):
    """
    Dedent according to indentation level of the last line.
    """
    lines = doc.splitlines()
    indent = SPACES.match(lines[-1]).end()
    return "\n".join(ln[indent:] for ln in lines)
