from pandas.testing import assert_series_equal, assert_frame_equal
from pytest import approx

import mundi
from pydemic.models import SIR
from pydemic.utils import flatten_dict


class TestInfoMixin:
    def test_model_to_json(self):
        m = SIR()

    def _test_regional_model_to_json(self):
        m = SIR(region="BR")
        br = mundi.region("BR")
        assert m.info.to_dict() == {
            "demography.population": br.population,
            "demography.age_distribution": br.age_distribution,
            "demography.age_pyramid": br.age_pyramid,
        }
        assert m.info.to_dict(flat=True) == flatten_dict(m.info.to_dict())


class TestRegionMixin:
    def test_initialization_options(self):
        br = mundi.region("BR")
        it = mundi.region("IT")

        # Initialize with no region
        m = SIR()
        assert m.population == 1_000_000
        assert m.region is None
        assert m.age_distribution is None
        assert m.age_pyramid is None

        # Use a region
        m = SIR(region="BR")
        assert m.population == br.population
        assert_series_equal(m.age_distribution, br.age_distribution, check_names=False)
        assert_frame_equal(m.age_pyramid, br.age_pyramid)

        # Mix parameters a region
        tol = 1e-6
        m = SIR(region="BR", population=1000)
        assert m.population == 1000
        assert abs(m.age_distribution.sum() - 1000) < tol
        assert abs(m.age_pyramid.sum().sum() - 1000) < tol

        ratio = br.age_distribution / m.age_distribution
        assert ((ratio - br.population / 1000).dropna().abs() < tol).all()

        # Mixed values: brazilian population with the age_distribution proportions
        # from Italy
        m = SIR(region="BR", age_distribution="IT")
        assert m.population == br.population
        assert m.age_distribution.sum() == approx(br.population)
        assert list(m.age_distribution / m.population) == approx(
            it.age_distribution / it.population
        )
