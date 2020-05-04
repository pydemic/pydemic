import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pydemic.diseases import covid19
from pydemic.diseases.covid19_api import epidemic_curve


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


class TestCovid19APIs:
    @pytest.mark.slow
    @pytest.mark.external
    def test_corona_api(self):
        br = epidemic_curve("BR", api="corona-api.com")
        assert list(br.columns) == ["cases", "fatalities", "recovered"]

    @pytest.mark.slow
    @pytest.mark.external
    def test_brasil_io_api(self):
        df = epidemic_curve("BR-DF", api="brasil.io")
        assert list(df.columns) == ["cases", "fatalities"]
