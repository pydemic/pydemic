import sidekick as _sk

from .datetime import today, now
from .formatting import fmt, pc, p10k, p100k, pm, indent, slugify, safe_int
from .functions import interpolant, lru_safe_cache, coalesce
from .json import to_json
from .properties import (
    state_property,
    param_property,
    param_transform,
    param_alias,
    inverse_transform,
)
from .sequences import rpartition, flatten_dict, unflatten_dict, extract_keys

not_implemented = lambda *args: _sk.error(NotImplementedError)
