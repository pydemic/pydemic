from sidekick.cache import ttl_cache, period_cache, set_global_memory_provider
from . import config

set_global_memory_provider(config.memory)
