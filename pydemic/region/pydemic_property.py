from .base import RegionProperty
from ..diseases import disease as get_disease


class PydemicProperty(RegionProperty):
    """
    Implements the "pydemic" property mokey-patched into mundi region objects.
    """

    __slots__ = ()

    def epidemic_curves(self, disease=None, **kwargs):
        """
        Return epidemic curves for region.
        """
        disease = get_disease(disease)
        return disease.epidemic_curve(self.region, **kwargs)

    def disease_params(self, disease=None, **kwargs):
        """
        Return an object with all disease params associated with region.
        """
        disease = get_disease(disease)
        return disease.params(region=self.region, **kwargs)
