import numpy as np
from pytest import approx
from pydemic.diseases import covid19
from pydemic.models import eSIR


class TestSIR:
    def test_basic_esir_api(self):
        m = eSIR(disease=covid19)
        m.run(30)
        res = m["I"]
        i0 = m.data.loc[m.times[0], "infectious"]
        ok = i0 * np.exp(m.K * m.times)

        assert m.R0 == 2.74
        assert m.gamma == approx(1 / m.infectious_period)
        assert m.K == approx(m.gamma * (m.R0 - 1))
        assert m.iter == len(m.data) == len(m.times) == len(m.dates)
        assert np.abs(res / ok - 1).max() < 1e-4
