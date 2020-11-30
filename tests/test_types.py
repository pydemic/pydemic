import pytest

from pydemic.params.model_params import SIRParams, SEIRParams, SEAIRParams
from pydemic.types import ComputedDict, DelayedArgsComputedDict, Result, inverse, transform, alias


class TestComputedDict:
    @pytest.fixture(scope="class")
    def SIR(self):
        class SIR(ComputedDict):
            R0: float = 1.0
            infectious_period: float = 0.5
            gamma: float = inverse("infectious_period")
            beta: float = transform(lambda R0, gamma: R0 * gamma)

        return SIR

    def test_computed_dict(self):
        dic = ComputedDict(
            R0=2.0,
            infectious_period=4,
            beta=lambda R0, gamma: R0 * gamma,
            gamma=lambda infectious_period: 1 / infectious_period,
            K=lambda R0, gamma: (R0 - 1) * gamma,
        )

        assert dic["R0"] == 2.0
        assert dic["infectious_period"] == 4.0
        assert dic["gamma"] == 0.25
        assert dic["K"] == 0.25
        assert dic["beta"] == 0.5
        assert dic.get_keys(["R0", "beta"]) == {"R0": 2.0, "beta": 0.5}

    def test_computed_dict_subclass(self, SIR):
        assert SIR()["R0"] == 1.0
        assert SIR().R0 == 1.0
        assert SIR(R0=2.0)["R0"] == 2.0
        assert SIR(R0=2.0).R0 == 2.0

        d = SIR()
        assert d["gamma"] == 2.0
        assert d["beta"] == 2.0
        assert d.R0 == 1.0
        assert d.beta == 2.0

        d = SIR(R0=2.0, beta=3.0)
        assert d["gamma"] == 2.0
        assert d["beta"] == 3.0
        assert d["R0"] == 2.0

    def test_delayed_computed_dict(self):
        d = DelayedArgsComputedDict({"t", "time"})
        d["x"] = 2
        d["y"] = lambda x: x ** 2
        d["f"] = lambda t: 2 * t

        assert d["x"] == 2
        assert d["y"] == 4
        assert callable(d["f"])
        assert d["f"](21) == 42

    def test_computed_dict_inverse_function(self, SIR):
        d = SIR()
        d["gamma"] = 1.0
        assert d["gamma"] == 1.0
        assert d["infectious_period"] == 1.0

        d["infectious_period"] = 2.0
        assert d["gamma"] == 0.5
        assert d["infectious_period"] == 2.0

    def test_computed_dict_mapping_interface(self, SIR):
        d = SIR()
        assert set(d.keys()) == {"R0", "infectious_period"}
        assert set(d.items()) == {("R0", 1.0), ("infectious_period", 0.5)}
        assert set(d.values()) == {1, 0.5}

    def test_computed_dict_subtype(self, SIR):
        class Sub(SIR):
            rho: float = 1.0
            prob_symptoms: float = 1
            Qs: float = alias("prob_symptoms")

        d = Sub()
        assert d._initial["R0"] == 1.0
        assert d["R0"] == 1.0
        assert d["gamma"] == 2.0
        assert d["infectious_period"] == 0.5
        assert d["rho"] == 1.0
        assert d["Qs"] == 1.0
        assert d["prob_symptoms"] == 1.0

    def test_model_computed_dicts(self):
        assert set(SIRParams()) == {"R0", "gamma"}
        assert set(SEIRParams()) == {"R0", "gamma", "incubation_period"}
        assert set(SEAIRParams()) == {"R0", "gamma", "incubation_period", "rho", "prob_symptoms"}

    def test_copy_keep_type(self):
        d = SIRParams()
        assert type(d.copy()) == SIRParams
        assert d.copy() == d


class TestNumericTypes:
    def test_numeric_type_accept_numeric_operations(self):
        x = Result(40)
        assert x + 2 == 2 + x == 42
        assert x.info is None

        y = Result(42, "the answer")
        assert y.value == 42
        assert y.info == "the answer"
