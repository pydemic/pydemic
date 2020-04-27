from typing import Type

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from pydemic import models

MODEL_CLASSES = [
    models.eSIR,
    # models.eSEIR,
    # models.eSEAIR,
    models.SIR,
    models.SEIR,
    models.SEAIR,
]


@pytest.fixture(params=MODEL_CLASSES, scope='session')
def model_cls(request) -> Type[models.Model]:
    return request.param


@pytest.fixture
def model(model_cls) -> models.Model:
    return model_cls()


class TestModels:
    @pytest.fixture
    def m(self, model):
        return model

    @pytest.fixture(scope='session')
    def cls(self, model_cls):
        return model_cls

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

    def test_model_can_run_during_initialization(self, cls):
        m1 = cls()
        m1.run(10)
        m2 = cls(run=10)
        assert all(m1.data == m2.data)

    def test_model_data_uses_integer_based_index(self, model):
        assert model.data.index.dtype == int


class TestSIR:
    def test_basic_esir_api(self):
        m = models.eSIR()
        m.run(30)
        res = m["I"]
        ok = m.data.loc[m.times[0], 'infectious'] * np.exp(m.K * m.times)

        assert m.R0 == 2.74
        assert abs(m.K - m.gamma * 1.74) <= 1e-6
        assert m.iter == len(m.data) == len(m.times) == len(m.dates)
        assert np.abs(res / ok - 1).max() < 1e-4
