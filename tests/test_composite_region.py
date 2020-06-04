from pandas.testing import assert_series_equal

from mundi import region
from pydemic.region.multi_region import CompositeRegion


class TestCompositeRegion:
    def test_composite_region_with_single_element(self):
        bsb = region("BR-5300108")
        reg = CompositeRegion([bsb])
        for attr in ("population", "name", "type", "subtype"):
            assert getattr(bsb, attr) == getattr(reg, attr)

    def test_arbitrary_composite_region(self):
        bsb = region("BR-5300108")
        sp = region("BR-3550308")
        reg = CompositeRegion([bsb, sp])

        assert reg.population == bsb.population + sp.population
        assert_series_equal(reg.age_distribution, bsb.age_distribution + sp.age_distribution)
