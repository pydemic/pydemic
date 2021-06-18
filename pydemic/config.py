from functools import lru_cache
from pathlib import Path

import joblib
import sidekick.api as sk

CACHE_OPTIONS = {}


@sk.once
def user_path():
    """
    Return the user path for pydemic cache and configuration files.
    """
    path = Path("~") / ".local" / "pydemic"
    path = path.expanduser()

    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path


@lru_cache(None)
def memory(name) -> joblib.Memory:
    """
    Return the joblib's Memory object with the given name.
    """
    if isinstance(name, joblib.Memory):
        return name

    path = user_path() / "cache" / name
    path.mkdir(parents=True, exist_ok=True)
    opts = CACHE_OPTIONS.get(name, {})
    opts.setdefault("verbose", 0)
    return joblib.Memory(path, **opts)


def set_cache_options(name, **kwargs):
    """
    Set options for the given cache.
    """
    if name in CACHE_OPTIONS and kwargs != CACHE_OPTIONS[name]:
        raise RuntimeError(f"cannot change caching options for {name}")
    CACHE_OPTIONS[name] = kwargs
