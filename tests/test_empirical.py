import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal

from pydemic.empirical import EpidemicCurve


class TestEpidemicCurve:
    def linearly_increasing(self):
        """
        Cases and deaths data with linearly increasing cases/deaths
        """
        duration = 61
        dates = pd.to_datetime(np.arange(1, duration), unit="d", origin="2020-01-01")
        deaths = np.maximum(np.arange(1, duration) - 5, 0)
        cases = np.arange(1, duration) * 10
        df = pd.DataFrame([cases, deaths], index=["cases", "deaths"], columns=dates).T
        params = {"population": 1_000_000, "case_fatality_ratio": 0.01, "infectious_period": 10.5}
        return EpidemicCurve(df, params)

    def exponentially_increasing(self, k=0.1):
        """
        Cases and deaths data with exponentially increasing cases/deaths
        """
        dates = pd.to_datetime(np.arange(1, 31), unit="d", origin="2020-01-01")
        deaths = np.maximum(np.exp(k * np.arange(1, 31)) - 5, 0)
        cases = np.exp(k * np.arange(1, 31)) * 10
        df = pd.DataFrame([cases, deaths], index=["cases", "deaths"], columns=dates).T
        params = {"population": 1_000_000, "case_fatality_ratio": 0.01, "infectious_period": 10.5}
        return EpidemicCurve(df, params)

    def test_epidemic_curves_on_linearly_increasing_data(self):
        data = self.linearly_increasing()
        assert_all_eq(data.new_cases(), 10)
        assert_all_eq(data.ascertaiment_rate(), 1.0)

        for p in [7, 14, 21]:
            incidence = data.incidence_rate(p)
            assert_all_eq(incidence.iloc[p:], 10 * p / 1e6)

        assert (data.point_prevalence() >= data.new_cases()).all()
        assert_array_almost_equal(data.period_prevalence(1), data.point_prevalence())


def assert_all_eq(lhs, scalar):
    if not (lhs == scalar).all():
        raise AssertionError(f"not all values equal to {scalar}: {lhs}")


def assert_all_simeq(lhs, scalar, tol=None):
    if tol is None:
        tol = 0.01 * scalar

    if not (abs(lhs - scalar) < tol).all():
        raise AssertionError(f"not all values similar to {scalar}: {lhs}")
