from typing import Type

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from pydemic import clinical_models
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
        assert isinstance(h, clinical_models.CrudeFR)
        assert type(h) == type(m.clinical.crude())
        assert isinstance(
            m.clinical.hospitalization_with_delay(), clinical_models.HospitalizationWithDelay
        )

    def test_clinical_model_basic_api(self, m):
        m.run(10)
        cm = m.clinical()
        c = cm["cases"]
        d = cm["deaths"]
        h = cm["hospitalized"]
        H = cm["hospitalizations"]
        assert all(h <= H)
        assert all(d <= H)
        assert all(H <= c)


class TestSIR:
    def test_basic_esir_api(self):
        m = models.eSIR()
        m.run(30)
        res = m["I"]
        ok = m.data.loc[m.times[0], "infectious"] * np.exp(m.K * m.times)

        assert m.R0 == 2.74
        assert abs(m.K - m.gamma * 1.74) <= 1e-6
        assert m.iter == len(m.data) == len(m.times) == len(m.dates)
        assert np.abs(res / ok - 1).max() < 1e-4
