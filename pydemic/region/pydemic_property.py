import pandas as pd

from .base import RegionProperty
from ..diseases import disease as get_disease, DiseaseParams


class PydemicProperty(RegionProperty):
    """
    Implements the "pydemic" property mokey-patched into mundi region objects.
    """

    __slots__ = ()

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
