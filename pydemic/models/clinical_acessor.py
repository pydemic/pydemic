import sidekick as sk

cm = sk.import_later("..clinical_models", package=__package__)


class Clinical:
    __slots__ = ("_model",)

    def __init__(self, model):
        self._model = model

    def __call__(self, *args, **kwargs):
        cls = self._model.clinical_model or cm.CrudeFR
        params = {**self._model.clinical_params, **kwargs}
        return self.clinical_model(cls, *args, **params)

    def clinical_model(self, cls, *args, **kwargs):
        """
        Create a clinical model from model infectious model instance.
        """
        return cls(self._model, *args, **kwargs)

    def crude_model(self, *args, **kwargs):
        """
        Create a clinical model from model infectious model instance.
        """
        return self.clinical_model(cm.CrudeFR, *args, **kwargs)

    def delay_model(self, *args, **kwargs):
        """
        A simple clinical model in which hospitalization occurs with some
        fixed delay.
        """
        return self.clinical_model(cm.HospitalizationWithDelay, *args, **kwargs)

    def overflow_model(self, *args, **kwargs):
        """
        A clinical model that considers the overflow of a healthcare system
        in order to compute the total death toll.
        """
        return self.clinical_model(cm.HospitalizationWithOverload, *args, **kwargs)
