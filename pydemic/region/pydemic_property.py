import pandas as pd

from .base import RegionProperty
from .. import fitting as fit
from ..diseases import disease as get_disease, DiseaseParams
from ..properties.decorators import cached
from ..types import ValueStd


class PydemicProperty(RegionProperty):
    """
    Implements the "pydemic" property mokey-patched into mundi region objects.
    """

    __slots__ = ()

    @cached(ttl="epidemic_curve")
    def epidemic_curve(self, disease=None, **kwargs) -> pd.DataFrame:
        """
        Return epidemic curves for region.

        See Also:
            :method:`pydemic.diseases.Disease.epidemic_curve`
        """
        disease = get_disease(disease)
        return disease.epidemic_curve(self.region, **kwargs)

    def disease_params(self, disease=None, **kwargs) -> DiseaseParams:
        """
        Return an object with all disease params associated with region.
        """
        disease = get_disease(disease)
        return disease.params(region=self.region, **kwargs)

    @cached(ttl="epidemic_curve")
    def estimate_Rt(self, model, disease=None, **kwargs) -> pd.DataFrame:
        """
        Compute R(t) from the epidemic curves. This is function is just a
        shortcut to :func:`fit.estimate_Kt` that automatically binds to the
        current region.
        """
        return self._estimate_R(fit.estimate_Rt, model, disease, **kwargs)

    @cached(ttl="epidemic_curve")
    def estimate_R0(self, model, disease=None, **kwargs) -> ValueStd:
        """
        Compute R0 from the epidemic curves. This is function is just a
        shortcut to :func:`fit.estimate_Kt` that automatically binds to the
        current region.
        """
        return self._estimate_R(fit.estimate_R0, model, disease, **kwargs)

    def _estimate_R(self, func, model, disease, **kwargs):
        disease = get_disease(disease)
        curves = self.epidemic_curve(disease)
        if isinstance(model, str) and "params" not in kwargs:
            kwargs["params"] = disease.params(region=self.region)
        return func(model, curves, **kwargs)

    @cached(ttl="epidemic_curve")
    def estimate_Kt(self, disease=None, **kwargs) -> pd.DataFrame:
        """
        Compute K(t) from the epidemic curves. This is function is just a
        shortcut to :func:`fit.estimate_Kt` that automatically binds to the
        current region.
        """
        return self._estimate_K(fit.estimate_Kt, disease, **kwargs)

    @cached(ttl="epidemic_curve")
    def estimate_K(self, disease=None, **kwargs) -> ValueStd:
        """
        Compute K from the epidemic curves. This is function is just a
        shortcut to :func:`fit.estimate_K` that automatically binds to the
        current region.
        """
        return self._estimate_K(fit.estimate_K, disease, **kwargs)

    def _estimate_K(self, func, disease, **kwargs):
        curves = self.epidemic_curve(disease)
        return func(curves, **kwargs)
