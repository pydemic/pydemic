from pydemic import models, ModelGroup
from pydemic.clinical_models import CrudeFR


class TestModelGroup:
    def test_model_group_from_split(self):
        m = models.SIR(R0=2, gamma=1)
        m.run(10)
        grp: ModelGroup = m.split(R0=[1.25, 1.5, 2.0, 2.5], name="R0 = {R0}")

        # Basic properties
        assert len(grp) == 4
        assert grp[0].name == "R0 = 1.25"
        assert grp[0].R0 == 1.25

        # Data
        assert grp["I"].shape == (11, 4)
        assert grp["I", 5:].shape == (6, 4)
        assert grp[["I", "R"], 5:].shape == (6, 8)

        # Clinical
        assert all(isinstance(cm, CrudeFR) for cm in grp.clinical.crude_model())
        assert all(isinstance(cm, CrudeFR) for cm in grp.clinical())
