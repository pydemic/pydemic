from pydemic.models import SIR


class TestModelMeta:
    def test_sir_model_meta_info(self):
        assert SIR._meta.params.primary == {"infectious_period", "R0"}
        # assert SIR._meta.params.derived == {"gamma", "beta", "K", "duplication_time"}
        assert SIR._meta.params.alternative == {"gamma"}
        # assert SIR._meta.params.alternative_map == {
        #     "gamma": "infectious_period",
        # }

    def _test_sir_model_instance_meta_info(self):
        m = SIR()
        assert m._meta.params.is_static("R0", m) is True
        assert m._meta.params.is_static("gamma", m) is True
        assert m._meta.params.is_static("beta", m) is True
        assert m._meta.params.is_static("infectious_period", m) is True
