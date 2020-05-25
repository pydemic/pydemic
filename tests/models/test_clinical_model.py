import mundi
from pydemic.diseases import covid19
from pydemic.models import SEIR


class TestClinicalModel:
    def test_clinical_model_uses_region_IFR(self, region="BR"):
        m = SEIR(region=region, disease=covid19)
        br = mundi.region("BR")
        m1 = m.clinical.crude_model()
        m2 = m.clinical.delay_model()
        m3 = m.clinical.overflow_model()

        IFR = covid19.IFR(region=region)
        CFR = covid19.CFR(region=region)

        for m in [m1, m2, m3]:
            assert m.region == br
            assert m.IFR == m.infection_fatality_ratio == IFR
            assert m.CFR == m.case_fatality_ratio == CFR
