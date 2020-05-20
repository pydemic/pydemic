import pytest
import sidekick as sk
from numpy.testing import assert_almost_equal
from sidekick import X, Y

from pydemic.formulas import sir, seir, seair, formula


class TestFormulaDecorator:
    def test_positional(self):
        @formula((), positional=1)
        def fn(f, gamma, sigma):
            return f(gamma, sigma)

        assert fn((X + Y), gamma=1, sigma=2) == 3
        assert fn((X + Y), {"gamma": 1, "sigma": 2}) == 3
        assert fn((X + Y), {"gamma": 1, "sigma": 2, "R0": 3}) == 3
        assert fn((X + Y), {"gamma": 1, "incubation_period": 0.5, "beta": 3}) == 3

        with pytest.raises(TypeError):
            fn((X + Y), gamma=1)

        with pytest.raises(TypeError):
            fn((X + Y), gamma=1, sigma=2, beta=3)

    def test_options(self):
        @formula((), positional=1, options=("power",))
        def fn(f, gamma, sigma, power=1):
            return f(gamma, sigma) ** power

        assert fn((X + Y), gamma=1, sigma=2) == 3
        assert fn((X + Y), gamma=1, sigma=2, power=2) == 9

    def test_ignore_kwargs(self):
        @formula((), invalid="pass")
        def fn(gamma, **kwargs):
            return gamma + sum(kwargs.values())

        assert fn(gamma=1, sigma=2) == 3
        assert fn(gamma=1, sigma=2, beta=3) == 6


class TestFormulaSIR:
    model = sir
    K = sk.delegate_to("model")
    R0 = sk.delegate_to("model")
    R0_from_K = sk.delegate_to("model")

    def test_R0(self):
        assert self.R0(beta=2, gamma=1) == 2
        assert self.R0.formula(beta=2, gamma=1) == 2
        assert self.R0.formula(2, 1) == 2
        assert self.R0(beta=2, infectious_period=2) == 4
        assert self.R0({"beta": 2, "gamma": 1}) == 2
        assert self.R0({"beta": 2, "gamma": 1}, gamma=2) == 1

    def test_R0_from_K(self):
        assert self.R0_from_K(K=0, gamma=1) == 1.0
        assert self.R0_from_K(K=0, gamma=2) == 1.0
        assert self.R0_from_K(K=1, gamma=2) == 1.5
        assert self.R0_from_K(K=2, gamma=2) == 2.0

    def test_K(self):
        assert self.K(R0=2, gamma=2) == 2.0
        assert self.K(R0=2, gamma=1) == 1.0
        assert self.K(R0=1, gamma=2) == 0.0
        assert self.K(R0=1, gamma=3) == 0.0
        assert self.K(R0=0.5, gamma=2) == -1.0
        assert self.K(R0=0.5, gamma=1) == -0.5

    def test_initial_state(self):
        assert_almost_equal(
            self.model.state_from_cases(population=10_000, cases=1000, R0=2, gamma=0.25),
            [9000, 333 + 2 / 3, 666 + 1 / 3],
        )


class TestFormulaSEIR(TestFormulaSIR):
    model = seir

    def test_K(self):
        assert self.K(R0=4, gamma=1, sigma=1) == 1.0
        assert self.K(R0=1, gamma=2, sigma=1) == 0.0
        assert self.K(R0=1, gamma=3, sigma=1) == 0.0
        assert self.K(R0=1 / 4, gamma=1, sigma=1) == -0.5

    def test_R0_from_K(self):
        assert self.R0_from_K(K=0, gamma=1, sigma=1) == 1.0
        assert self.R0_from_K(K=0, gamma=2, sigma=1) == 1.0
        assert self.R0_from_K(K=1, gamma=1, sigma=1) == 4.0
        assert self.R0_from_K(K=2, gamma=1, sigma=1) == 9.0

    def test_initial_state(self):
        assert_almost_equal(
            self.model.state_from_cases(
                population=10_000, cases=1000, R0=2, gamma=0.25, sigma=0.25
            ),
            [8705.9854608, 294.0145392, 207.8996744, 792.1003256],
        )


class TestFormulaSEAIR(TestFormulaSEIR):
    model = seair

    def test_R0(self):
        assert self.R0(beta=2, gamma=1, prob_symptoms=1.0, rho=0.25) == 2
        assert self.R0(beta=2, gamma=1, prob_symptoms=1.0, rho=0.5) == 2

        assert self.R0(beta=2, gamma=1, prob_symptoms=0.0, rho=0.25) == 0.5

        assert self.R0(beta=2, gamma=1, prob_symptoms=0.5, rho=1.0) == 2
        assert self.R0(beta=2, gamma=1, prob_symptoms=0.25, rho=1.0) == 2

        assert self.R0(beta=2, gamma=1, prob_symptoms=0.25, rho=0.0) == 0.5
        assert self.R0(beta=2, gamma=1, prob_symptoms=0.5, rho=0.0) == 1.0

        assert self.R0(beta=2, gamma=1, prob_symptoms=0.5, rho=0.5) == 1.5

    def test_initial_state(self):
        assert_almost_equal(
            self.model.state_from_cases(
                population=10_000,
                cases=1000,
                R0=2,
                gamma=0.25,
                sigma=0.25,
                prob_symptoms=0.5,
                rho=0.5,
            ),
            [9204.0712473, 588.0290783, 207.8996744, 207.8996744, 1584.2006512],
        )
