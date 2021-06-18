import math
from tokenize import Number, Name

import numpy as np
from lark import Lark, InlineTransformer, LarkError
from sidekick import api as sk


class Function:
    """
    A picklable function-like object.
    """

    argnames = sk.delegate_to("_impl")
    signature = sk.delegate_to("_impl")
    __signature__ = sk.delegate_to("_impl")
    __wrapped__ = sk.alias("_impl")

    def __init__(self, src, env=None):
        self.src = src
        self.env = env
        self._impl = compile_expr(src, env)

    def __call__(self, *args, **kwargs):
        return self._impl(*args, **kwargs)

    def __getstate__(self):
        return self.src, self.env

    def __setstate__(self, state):
        self.src, self.env = state
        self._impl = compíle_expr(self.src, self.env)


def inverse(attr):
    """
    Declare field as an inverse transform of another field.
    """
    fn = eval(f"lambda {attr}: 1 / {attr}")
    fn._func_inverse_ = lambda x: 1 / x
    return fn


def transform(fn, inv=None, args=None):
    """
    Declare field as a transform of other fields.
    """
    if isinstance(fn, str):
        fn = Function(fn)
    if isinstance(inv, str):
        inv = Function(inv)

    fn = sk.to_callable(fn)
    if inv is not None:
        fn._func_inverse_ = inv
    if args is not None:
        fn.argnames = args
    fn.computed_key = True
    return fn


def statictransform(fn, inv=None):
    """
    Declares a static method as a transform.

    This is not necessary, but exists to keep static analyzers happier.
    """
    return staticmethod(transform(fn, inv))


def alias(attr):
    """
    Declare field as an alias of another field.
    """
    fn = eval(f"lambda {attr}: {attr}")
    fn._func_inverse_ = lambda x: x
    return fn


def initial_values(cls):
    """
    Get dictionary of initial value mappings from base classes.
    """
    initial = {}
    for b in reversed(cls.__bases__):
        initial.update(getattr(b, "_initial", None) or ())
    return initial


def compile_expr(src, env=None):
    """
    Compile simple mathematically oriented expressions into lambda functions.

    Examples:
        >>> f = compile_expr("X * Y")
        >>> f(2, 3)
        6
        >>> g = compile_expr("sqrt(x)")
        >>> g(4)
        2.0
    """
    env = env if env is not None else ENV
    try:
        tree = grammar.parse(src)
    except LarkError as ex:
        raise SyntaxError(f"invalid expression {src!r}:\n{ex}")
    transformer = CalcTransformer()
    transformer.transform(tree)

    ns = {k: env[k] for k in transformer.funcnames}
    varnames = transformer.varnames[:]
    for var in transformer.varnames:
        if var in env:
            ns[var] = env[var]
            varnames.remove(var)

    args = ", ".join(varnames)
    fn = eval(f"lambda {args}: {src}", ns)
    fn.argnames = varnames
    fn.source = src
    return fn


#
# Utility functions
#
def get_computed_key_declarations(cls) -> set:
    """
    Return a dictionary of computed key declarations.
    """
    hints = set(getattr(cls, "__annotations__", None) or ())
    items = cls.__dict__.items()
    hints.update(k for k, v in items if getattr(v, "computed_key", False))
    return set(hints)


def get_argnames(fn):
    """
    Get a tuple with function's required positional argument names.
    """
    try:
        return fn.argnames
    except AttributeError:
        return sk.signature(fn).argnames()


def partial_args(argnames, args, kwargs):
    """
    Return a new tuple of argnames as if it has partially applied some
    positional and keyword arguments.
    """
    argnames = argnames[: len(args)]
    return tuple(filter(lambda x: x not in kwargs, argnames))


#
# Expression grammar
#

ENV = {**vars(math), "np": np, "math": math}

grammar = Lark(
    rf"""
?start : expr

?expr  : expr /[+-]/ term
       | term

?term  : term /[*\/%@]/ pow
       | /[+-]/ term
       | pow

?pow   : atom "**" pow
       | atom

atom   : NUMBER
       | NAME               -> name
       | "(" expr ")"
       | fname "(" [args] ")"

args   : expr ("," expr)*
fname  : NAME ("." NAME)*

NUMBER.1 : /{Number}/
NAME.0   : /{Name}/

%ignore /\s+/
""",
    parser="lalr",
)


class CalcTransformer(InlineTransformer):
    def __init__(self):
        super().__init__()
        self.varnames = []
        self.funcnames = set()

    def name(self, name):
        self.varnames.append(str(name))
        return name

    def fname(self, base, *args):
        self.funcnames.add(base)
