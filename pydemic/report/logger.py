import os
from typing import Type

import sidekick.api as sk

LOG_LEVELS = frozenset({"debug", "info", "warning", "error", "critical"})
SYSTEM_DEBUG = os.environ.get("DEBUG", "").lower() == "true"
DEFAULT_LEVEL = os.environ.get("PYDEMIC_LOG_LEVEL", "debug" if SYSTEM_DEBUG else "info")


class Logger:
    """
    Log messages and errors.

    When an error occurs, logger enters an error state with a falsy value.
    """

    __slots__ = ("_handler", "_level", "_error")
    has_error = property(lambda self: self._error is not None)
    error_code = sk.alias("_error")

    def __init__(self, handler=print, level=DEFAULT_LEVEL):
        self._handler = handler
        self._level = check_level(level)
        self._error = None

    def __call__(self, *args, level=None, **kwargs):
        level = check_level(level or self._level)
        return getattr(self, level)(*args, **kwargs)

    def __bool__(self):
        return self._error is None

    def clear(self):
        """
        Clear errors.
        """
        self._error = None

    def check_errors(self, exception: Type[Exception] = None):
        """
        Raises error if logger is in an error state.
        """
        error = self._error
        if error is None:
            return
        elif exception is None:
            if (
                isinstance(error, Exception)
                or isinstance(error, type)
                and issubclass(error, Exception)
            ):
                raise error
            else:
                raise ValueError(error)
        else:
            raise Exception(error)

    def apply(*args, **kwargs):
        """
        Run function with given arguments if not in error state.
        """
        self, func, *args = args
        if self:
            return func(*args, **kwargs)

    def debug(self, msg):
        """
        Print a debug log message.
        """
        self._handler(f"[debug] {msg}")
        return self

    def info(self, msg):
        """
        Print an info log message.
        """
        self._handler(f"[info] {msg}")
        return self

    def warning(self, msg):
        """
        Print an warning log message.

        Warnings are issued to unexpected conditions.
        """
        self._handler(f"[warning] {msg}")
        return self

    def error(self, msg, code=None):
        """
        Print an error log message and mark logger with the given error.
        """
        self._handler(f"[error] {msg}")
        code = msg if code is None else code
        self._error = code
        return self

    def critical(self, msg, code=None):
        """
        Print a critical log message and mark logger with the given error.

        Critical messages raise a SystemExit exception that terminates program
        execution if not handled by the callee.
        """
        self._handler(f"[error] {msg}")
        code = msg if code is None else code
        self._error = code
        raise SystemExit(code)


class HasLoggerMixin:
    """
    Mixin class for objects that expose a logger.

    The mixin implements a fluid interface
    """

    __slots__ = ("_logger",)
    _logger: Logger
    _log_value = property(lambda self: self)
    has_errors = sk.delegate_to("_logger")
    error_code = sk.delegate_to("_logger")

    def __init__(self, logger=None):
        if logger is None:
            logger = Logger()
        self._logger = logger

    def log(self, msg, level=None):
        """
        Log message using the default logging mechanism.
        """
        self._logger(msg, level=level)
        return self._log_value

    def log_debug(self, msg):
        self._logger.debug(msg)
        return self._log_value

    def log_info(self, msg):
        self._logger.info(msg)
        return self._log_value

    def log_warning(self, msg):
        self._logger.warning(msg)
        return self._log_value

    def log_error(self, msg, code=None):
        self._logger.error(msg, code)
        return self._log_value

    def log_critical(self, msg, code=None):
        self._logger.critical(msg, code)
        return self._log_value


def check_level(level):
    """
    Check if given level is valid and return a normalized copy.
    """
    if level is None:
        return DEFAULT_LEVEL
    if level not in LOG_LEVELS:
        raise ValueError(f"invalid debug level: {level!r}")
    return level
