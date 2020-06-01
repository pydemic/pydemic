# flake8: noqa
import sidekick as _sk

from .datetime import today, now
from .formatting import (
    fmt,
    pc,
    p10k,
    p100k,
    pm,
    indent,
    slugify,
    safe_int,
    format_args,
    file_type_display_name,
)
from .functions import interpolant, lru_safe_cache, coalesce, maybe_run
from .json import to_json
from .properties import (
    state_property,
    param_property,
    param_transform,
    param_alias,
    inverse_transform,
)
from .sequences import (
    rpartition,
    flatten_dict,
    unflatten_dict,
    extract_keys,
    sliced,
    as_seq,
    is_seq,
)
from .dataframe import trim_zeros
from .timeseries import accumulate_weekly, day_of_week, weekday_name, trim_weeks

not_implemented = lambda *args: _sk.error(NotImplementedError)
