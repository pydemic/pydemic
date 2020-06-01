from pydemic.models import SIR
from pydemic.models.abstract_sir import AbstractSIR


class TestMetaClass:
    def test_meta_class_for_derived_classes(self):
        assert AbstractSIR.meta.model_name == "SIR"
        assert SIR.meta.model_name == "SIR"

    def test_meta_class_introspection(self):
        meta = SIR.meta

        # State variables
        assert meta.variables == ("susceptible", "infectious", "recovered")
        assert meta.ndim == 3
        assert meta.data_aliases == {
            **dict(zip("SIR", meta.variables)),
            "E": "infectious",
            "exposed": "infectious",
        }

        # Parameters
        assert meta.params.primary == {"R0", "infectious_period"}
        assert meta.params.alternative == {"gamma"}
        assert meta.params.all == {*meta.params.primary, *meta.params.alternative}
