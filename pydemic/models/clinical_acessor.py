from typing import Union, TYPE_CHECKING

import sidekick as sk

if TYPE_CHECKING:
    from ..clinical_models import CrudeFR, HospitalizationWithDelay, HospitalizationWithOverflow

    ClinicalModel = Union["CrudeFR", "HospitalizationWithDelay", "HospitalizationWithOverflow"]


class Clinical:
    """
    Implements the ``.clinical`` attribute of models.
    """

    __slots__ = ("_model",)

    _default = sk.lazy(lambda x: x())
    info = sk.delegate_to("_default")
    params = sk.delegate_to("_default")
    summary = sk.delegate_to("_default")

    @property
    def _models(self):
        from .. import clinical_models

        return clinical_models

    def __init__(self, model):
        self._model = model

    def __call__(self, *args, **kwargs):
        params = {**self._model.clinical_params, **kwargs}
        if args:
            (cls,) = args
            if isinstance(cls, str):
                return self.clinical_model(cls, **params)
        else:
            cls = self._model.clinical_model or self._models.CrudeFR
        return self.clinical_model(cls, **params)

    def __getitem__(self, item):
        return self._default[item]

    def clinical_model(self, cls, **kwargs) -> "ClinicalModel":
        """
        Create a clinical model from model infectious model instance.
        """
        if isinstance(cls, str):
            method = getattr(self, cls.lower() + "_model", None)
            if method is None:
                raise ValueError(f"invalid model: {cls!r}")
            return method(**kwargs)

        return cls(self._model, **kwargs)

    def crude_model(self, **kwargs) -> "CrudeFR":
        """
        Create a clinical model from model infectious model instance.
        """
        cls = self._models.CrudeFR
        return self.clinical_model(cls, **kwargs)

    def delay_model(self, **kwargs) -> "HospitalizationWithDelay":
        """
        A simple clinical model in which hospitalization occurs with some
        fixed delay.
        """
        cls = self._models.HospitalizationWithDelay
        return self.clinical_model(cls, **kwargs)

    def overflow_model(self, **kwargs) -> "HospitalizationWithOverflow":
        """
        A clinical model that considers the overflow of a healthcare system
        in order to compute the total death toll.
        """
        cls = self._models.HospitalizationWithOverflow
        return self.clinical_model(cls, **kwargs)
