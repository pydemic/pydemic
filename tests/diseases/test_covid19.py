import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pytest import approx

from pydemic.api.covid19 import epidemic_curve
from pydemic.diseases import covid19


class TestCovid19:
    def test_covid19_reference_tables(self):
        for source in [None, "verity"]:
            assert isinstance(covid19.mortality_table(source), pd.DataFrame)
        for source in [None, "verity"]:
            assert isinstance(covid19.hospitalization_table(source), pd.DataFrame)

    def test_covid19_default_reference_tables(self):
        table = covid19.mortality_table
        assert_frame_equal(table(), table("verity"))

        table = covid19.hospitalization_table
        assert_frame_equal(table(), table("verity"))

    def test_covid_dict_params(self):
        data = covid19.to_dict()
        res = {
            "R0": 2.74,
            "rho": 0.45,
            #
            "case_fatality_ratio": 0.011,
            "infection_fatality_ratio": 0.00566,
            #
            "icu_fatality_ratio": 0.490,
            "hospital_fatality_ratio": 0.141,
            #
            "incubation_period": 3.69,
            "infectious_period": 3.47,
            #
            "hospitalization_period": 10.0,
            "icu_period": 7.5,
            #
            "critical_period": 7.5,
            "severe_period": 10.0,
            #
            "prob_critical": 0.0225,
            "prob_severe": 0.0778,
            "prob_symptoms": 0.532,
            #
            "hospitalization_overflow_bias": 0.25,
        }
        for k, v in res.items():
            assert data[k] == approx(v, rel=0.01, abs=1e-3), k
        assert set(res.keys()) == set(data.keys())
        assert data == covid19.to_json()

    def _test_params(self):
        p = covid19.params()

        # Aliases
        assert p.Qs == p.prob_symptoms
        assert p.Qsv == p.prob_severe
        assert p.Qcr == p.prob_critical

        # Transforms
        assert p.gamma == 1 / p.infectious_period
        assert p.sigma == 1 / p.incubation_period


class TestCovid19APIs:
    @pytest.mark.slow
    @pytest.mark.external
    def test_corona_api(self):
        br = epidemic_curve("BR", api="corona-api.com", extra=True)
        assert list(br.columns) == ["cases", "deaths", "recovered"]

        br = epidemic_curve("BR", api="corona-api.com", extra=False)
        assert list(br.columns) == ["cases", "deaths"]

    @pytest.mark.slow
    @pytest.mark.external
    def test_brasil_io_api(self):
        df = epidemic_curve("BR-DF", api="brasil.io")
        assert list(df.columns) == ["cases", "deaths"]
