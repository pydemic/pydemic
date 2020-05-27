import locale
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from pydemic import utils

DAY = timedelta(days=1)


class TestUtilityFunctions:
    def test_format_functions_en_US(self):
        fmt = utils.fmt

        try:
            locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
        except locale.Error:
            return pytest.skip()

        assert fmt(None) == "-"
        assert fmt(float("inf")) == "infinity"
        assert fmt(0.10) == "0.1"
        assert fmt(0.12) == "0.12"
        assert fmt(0.01) == "0.01"
        assert fmt(0.012) == "0.012"
        assert fmt(0.0123) == "0.012"
        assert fmt(0.00123) == "1.23e-03"
        assert fmt(0.0012) == "1.2e-03"
        assert fmt(1.2341) == "1.23"
        assert fmt(12.341) == "12.34"
        assert fmt(123.41) == "123.4"
        assert fmt(1234) == "1,234"
        assert fmt(1234.5) == "1,234"
        assert fmt(42_123.1) == "42,123"
        assert fmt(42_123) == "42,123"
        assert fmt(1_000_000) == "1M"
        assert fmt(10_000_000) == "10M"
        assert fmt(12_000_000) == "12M"
        assert fmt(12_300_000) == "12.3M"
        assert fmt(12_340_000) == "12.34M"
        assert fmt(12_341_000) == "12.34M"
        assert fmt(-12_341_000) == "-12.34M"
        assert fmt(123_456_000) == "123.5M"
        assert fmt(1_234_567_000) == "1.23B"

    def test_format_functions_pt_BR(self):
        fmt = utils.fmt
        try:
            locale.setlocale(locale.LC_ALL, "pt_BR.UTF-8")
        except locale.Error:
            return pytest.skip()

        assert fmt(None) == "-"
        assert fmt(0.10) == "0,1"
        assert fmt(0.12) == "0,12"
        assert fmt(0.01) == "0,01"
        assert fmt(0.012) == "0,012"
        assert fmt(0.0123) == "0,012"
        assert fmt(0.00123) == "1,23e-03"
        assert fmt(0.0012) == "1,2e-03"
        assert fmt(1.2341) == "1,23"
        assert fmt(12.341) == "12,34"
        assert fmt(123.41) == "123,4"
        assert fmt(1234) == "1.234"
        assert fmt(1234.5) == "1.234"
        assert fmt(42_123.1) == "42.123"
        assert fmt(42_123) == "42.123"
        assert fmt(1_000_000) == "1M"
        assert fmt(10_000_000) == "10M"
        assert fmt(12_000_000) == "12M"
        assert fmt(12_300_000) == "12,3M"
        assert fmt(12_340_000) == "12,34M"
        assert fmt(12_341_000) == "12,34M"
        assert fmt(-12_341_000) == "-12,34M"
        assert fmt(123_456_000) == "123,5M"
        assert fmt(1_234_567_000) == "1,23B"

    def test_other_formats(self):
        assert utils.pc(0.5) == "50%"
        assert utils.pm(0.05) == "50‰"
        assert utils.p10k(0.005) == "50‱"
        assert utils.p100k(0.0005) == "50/100k"
        assert utils.safe_int(3.14) == 3
        assert utils.safe_int(float("nan")) == 0

    def test_text_functions(self):
        assert utils.indent("foo\nbar", 2) == "  foo\n  bar"
        assert utils.slugify("Foo Bar") == "foo-bar"

    def test_format_args(self):
        assert utils.format_args(1, 2, op="sum") == "1, 2, op='sum'"


class TestDatetime:
    def test_now_and_today(self):
        assert utils.now().date() == utils.today()


class TestSequenceFunctions:
    def test_flatten_and_unflatten_dict(self):
        flat = {"foo.bar.bat": 2, "foo.bar.baz": 1, "foo.bot": 3}
        nested = {"foo": {"bar": {"baz": 1, "bat": 2}, "bot": 3}}
        assert utils.flatten_dict(nested) == flat
        assert utils.unflatten_dict(flat) == nested


class TestJson:
    def test_convert_to_json(self):
        for x in [True, None, "string", 42, 3.14, ["list", "of items"], {"dict": "too"}]:
            assert utils.to_json(x) == x

        assert utils.to_json({0: 0, 3.14: 1, True: 2, None: 3, ...: 4}) == {
            "0": 0,
            "3.14": 1,
            "true": 2,
            "null": 3,
            "...": 4,
        }

    def test_to_json_as_method(self):
        class Foo:
            def to_json(self):
                return {"type": "Foo"}

        foo = Foo()
        assert utils.to_json(foo) == {"type": "Foo"}

    def test_to_json_raises_a_type_error_in_unsupported_types(self):
        class Foo:
            pass

        foo = Foo()
        with pytest.raises(TypeError):
            utils.to_json(foo)

        with pytest.raises(TypeError):
            utils.to_json({(1, 2): 3})


class TestDataFrame:
    def test_trim_zeros(self):
        data1 = [0, 0, 0, 1, 2, 0, 3, 0, 0]
        data2 = pd.Series(data1)
        data3 = [1, *data1, 2]
        assert_array_equal(utils.trim_zeros(data1), [1, 2, 0, 3])
        assert_array_equal(utils.trim_zeros(data2), [1, 2, 0, 3])
        assert_array_equal(utils.trim_zeros(data3), data3)
        assert_array_equal(utils.trim_zeros([0, 0, 0]), [])


class TestFunctions:
    def test_maybe_run(self):
        from math import sqrt

        assert utils.maybe_run(sqrt, 4) == 2
        assert utils.maybe_run(sqrt, None) is None

    def test_coalesce(self):
        with pytest.raises(ValueError):
            utils.coalesce(None, None, raises=True)

        with pytest.raises(ValueError):
            utils.coalesce(raises=True)

        assert utils.coalesce() is None
        assert utils.coalesce(1, 2) == 1
        assert utils.coalesce(1, None) == 1
        assert utils.coalesce(None, 1) == 1

    def test_interpolant(self):
        x = [1, 2, 3]
        y = [1, 2, 4]
        fn = utils.interpolant(x, y)

        assert fn(1.5) == 1.5
        assert fn(2.5) == 3.0
        assert_array_equal(fn([1.5, 2.5]), [1.5, 3.0])

    def test_lru_safe_cache(self):
        compute = False

        @utils.lru_safe_cache(10)
        def fn(n):
            nonlocal compute
            compute = True
            return [n] * n

        assert fn(2) == [2, 2]
        assert compute is True

        compute = False
        assert fn(2) == [2, 2]
        assert compute is False

        assert fn(2) is not fn(2)


class TestTimeseries:
    def get_index(self, size, weekday=0):
        monday = pd.to_datetime("2020-05-25")
        start = monday + DAY * weekday
        return pd.to_datetime([start + n * DAY for n in range(size)])

    def get_series(self, size, weekday=0):
        index = self.get_index(size, weekday)
        return pd.Series(np.arange(size), index=index)

    def get_dataframe(self, size, weekday=0):
        index = self.get_index(size, weekday)
        return pd.DataFrame(
            {"even": np.arange(0, 2 * size, 2), "odd": np.arange(1, 2 * size, 2)}, index=index
        )

    def test_trim_weeks(self):
        data = self.get_dataframe(18)
        clean = utils.trim_weeks(data)
        assert len(clean) == 14
        assert_frame_equal(clean, data.iloc[:14])

        clean = utils.trim_weeks(data, 1)
        assert len(clean) == 14
        assert_frame_equal(clean, data.iloc[1:15])

    def test_accumulate_weekly(self):
        data = self.get_dataframe(25, 0)
        acc = utils.accumulate_weekly(data, "trim")
        assert_array_equal(data.iloc[0:7].sum(), acc.iloc[0])

        data = self.get_dataframe(25, 2)
        acc = utils.accumulate_weekly(data, "trim")
        assert_array_equal(data.iloc[5:12].sum(), acc.iloc[0])

    def test_day_of_week(self):
        data = 1

    def test_weekday_name(self):
        assert utils.weekday_name(0) == "Monday"
        assert_array_equal(utils.weekday_name([0, 2]), ["Monday", "Wednesday"])
