import math
from tokenize import Number, Name

import numpy as np
from lark import Lark, InlineTransformer, LarkError

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
