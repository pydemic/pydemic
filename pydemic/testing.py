import locale
import os

import pytest


@pytest.yield_fixture
def en():
    en = "en_US.UTF-8"
    lc_all = locale.getlocale(locale.LC_ALL)
    lc_messages = locale.getlocale(locale.LC_MESSAGES)
    lang = os.environ.get("LANG")
    language = os.environ.get("LANGUAGE")

    try:
        locale.setlocale(locale.LC_ALL, en)
        locale.setlocale(locale.LC_MESSAGES, en)
        os.environ["LANG"] = en
        os.environ["LANGUAGE"] = en
        yield en
    except locale.Error:
        return pytest.skip("en_US.UTF-8 locale is not available in this machine")
    finally:
        locale.setlocale(locale.LC_ALL, lc_all)
        locale.setlocale(locale.LC_MESSAGES, lc_messages)

    if lang:
        os.environ["LANG"] = lang
    else:
        del os.environ["LANG"]

    if language:
        os.environ["LANGUAGE"] = language
    else:
        del os.environ["LANGUAGE"]
