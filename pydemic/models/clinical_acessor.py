import sidekick as sk

cm = sk.import_later("..clinical_models", package=__package__)


class Clinical:
    __slots__ = ("_model",)

    def __init__(self, model):
        self._model = model

    def __call__(self, *args, **kwargs):
        cls = getattr(self._model, "clinical_model", cm.CrudeFR)
        return self.clinical_model(cls, *args, **kwargs)

    def clinical_model(self, cls, *args, **kwargs):
        """
        Create a clinical model from model infectious model instance.
        """
        return cls(self._model, *args, **kwargs)

    def crude(self, *args, **kwargs):
        """
        Create a clinical model from model infectious model instance.
        """
        return self.clinical_model(cm.CrudeFR, *args, **kwargs)

    def hospitalization_with_delay(self, *args, **kwargs):
        """
        A simple clinincal model in which hospitalization occurs with some
        fixed delay.
        """
        return self.clinical_model(cm.HospitalizationWithDelay, *args, **kwargs)
