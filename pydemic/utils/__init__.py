import sidekick as _sk

from .datetime import today, now
from .formatting import fmt, pc, p10k, p100k, pm, indent, slugify
from .functions import interpolant, lru_safe_cache
from .properties import (
    state_property,
    param_property,
    param_transform,
    param_alias,
    inverse_transform,
)
from .sequences import rpartition

not_implemented = lambda *args: _sk.error(NotImplementedError)
