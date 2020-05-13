import sidekick as sk


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
        cls = self._model.clinical_model or self._models.CrudeFR
        params = {**self._model.clinical_params, **kwargs}
        return self.clinical_model(cls, *args, **params)

    def __getitem__(self, item):
        return self._default[item]

    def clinical_model(self, cls, *args, **kwargs):
        """
        Create a clinical model from model infectious model instance.
        """
        return cls(self._model, *args, **kwargs)

    def crude_model(self, *args, **kwargs):
        """
        Create a clinical model from model infectious model instance.
        """
        cls = self._models.CrudeFR
        return self.clinical_model(cls, *args, **kwargs)

    def delay_model(self, *args, **kwargs):
        """
        A simple clinical model in which hospitalization occurs with some
        fixed delay.
        """
        cls = self._models.HospitalizationWithDelay
        return self.clinical_model(cls, *args, **kwargs)

    def overflow_model(self, *args, **kwargs):
        """
        A clinical model that considers the overflow of a healthcare system
        in order to compute the total death toll.
        """
        cls = self._models.HospitalizationWithOverflow
        return self.clinical_model(cls, *args, **kwargs)
