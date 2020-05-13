import re
from typing import Type

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from pydemic.clinical_models import CrudeFR, HospitalizationWithDelay
from pydemic import models

MODEL_CLASSES = [
    models.eSIR,
    # models.eSEIR,
    # models.eSEAIR,
    models.SIR,
    models.SEIR,
    models.SEAIR,
]


@pytest.fixture(params=MODEL_CLASSES, scope="session")
def model_cls(request) -> Type[models.Model]:
    return request.param


@pytest.fixture
def model(model_cls) -> models.Model:
    return model_cls()


class ModelTester:
    @pytest.fixture
    def m(self, model):
        return model

    @pytest.fixture(scope="session")
    def cls(self, model_cls):
        return model_cls


class TestInfectiousModels(ModelTester):
    def test_model_basic_api_interactions(self, m):
        m.run(7)
        assert m.R0 > 1
        assert m.name

        # Basic invariants
        assert m.iter == len(m.data)
        assert all(m.times == m.data.index)

        # Test the infectious column
        I = m["I"]
        assert_series_equal(I, m["infectious"])
        assert any(I > 0.0)
        assert I.dtype == float
        assert isinstance(I, pd.Series)
        assert all(m["I:dates"].index == m.dates)

        # Test item accessors
        assert m["I", 0] == m.data.loc[0, "infectious"]
        assert m["I", -1] == m.data.loc[len(m.data) - 1, "infectious"]

    def test_can_override_model_parameters_during_initialization(self, cls):
        m = cls(R0=2)
        assert m.R0 == 2.0

        m.run(3)
        assert m.R0 == 2.0

    def test_model_can_run_during_initialization(self, cls):
        m1 = cls()
        m1.run(10)
        m2 = cls(run=10)
        assert all(m1.data == m2.data)

    def test_model_time_is_not_reset_from_clinical_model(self, m):
        m.run(10)
        date = m.date
        h = m.clinical()
        assert h.time == m.time == 10
        assert h.date == m.date == date

    def test_model_can_run_multiple_times(self, m):
        m.run(5)
        m.run(5)
        m.run(5)
        assert m.time == 15
        assert len(m.times) == 16
        assert len(set(m.times)) == 16
        assert all(np.diff(m.times) > 0)

    def test_model_data_uses_numeric_index(self, m):
        assert m.data.index.dtype == float

    def test_model_exposes_parameters_as_timeseries(self, m):
        m.run(7)
        R0 = m.R0
        series = m["R0"]
        print(series)
        assert len(series) == 8
        assert (series == R0).all()

    def test_clinical_accessor(self, m):
        h = m.clinical()
        assert h.empirical_CFR < m.exposed
        assert h.empirical_IFR < m.exposed
        assert isinstance(h, CrudeFR)
        assert type(h) == type(m.clinical.crude_model())
        assert isinstance(m.clinical.delay_model(), HospitalizationWithDelay)

    def test_clinical_model_basic_api(self, m):
        m.run(10)
        cm = m.clinical()
        c = cm["cases"]
        d = cm["deaths"]
        h = cm["hospitalized"]
        H = cm["hospitalized_cases"]
        assert all(h <= H)
        assert all(d <= H)
        assert all(H <= c)


class TestGetitemInteface:
    def test_transforms(self):
        m = models.SIR(R0=3, infectious_period=0.5, run=10, date="1970-01-01")

        I = m["I"]
        first, last = I.iloc[[0, -1]]
        assert_series_equal(m["infectious"], I)
        assert_series_equal(m["infectious:int"], m["I:int"])

        # Type conversions
        assert m["I:int"].dtype == int
        assert m["I:round"].dtype == int
        assert m["I:int:max"] >= 250_000
        assert_pattern(r"\d+", m["I:round"])
        assert_pattern(r"\d+\.\d", m["I:round1"])
        assert_pattern(r"\d+\.\d{1,2}", m["I:round2"])
        assert_pattern(r"\d+\.\d{1,3}", m["I:round3"])

        # Simple transforms
        assert m["I:initial"] == first
        assert m["I:final"] == last
        assert m["I:max"] >= 250_000
        assert m["I:min"] == 1.0
        assert m["I:peak-time"] == 4.0
        assert m["I:peak-date"] == pd.to_datetime("1970-01-05")
        assert type(m["I:np"]) is np.ndarray
        assert m["I:np:min"] == 1.0
        assert m["I:str:initial"] == "1.0"

        # Rate conversions
        assert (m["I:pp"] <= 1).all()
        assert (m["I:ppc"] <= 100).all()
        assert (m["I:p1k"] <= 1e3).all()
        assert (m["I:p10k"] <= 1e4).all()
        assert (m["I:p100k"] <= 1e5).all()
        assert (m["I:p1m"] <= 1e6).all()
        assert m["I:ppc:max"] >= 25


def assert_pattern(regex, data):
    pattern = re.compile(regex)
    check = pattern.fullmatch
    for i, x in enumerate(data):
        if check(str(x)) is None:
            msg = f"invalid pattern found in pos {i}: {x!r}"
            raise AssertionError(msg)
