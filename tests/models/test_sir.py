import numpy as np

from pydemic.diseases import covid19
from pydemic.models import eSIR


class TestSIR:
    def test_basic_esir_api(self):
        m = eSIR(disease=covid19)
        m.run(30)
        res = m["I"]
        ok = m.data.loc[m.times[0], "infectious"] * np.exp(m.K * m.times)

        assert m.R0 == 2.74
        assert abs(m.K - m.gamma * 1.74) <= 1e-6
        assert m.iter == len(m.data) == len(m.times) == len(m.dates)
        assert np.abs(res / ok - 1).max() < 1e-4
