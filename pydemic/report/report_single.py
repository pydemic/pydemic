from typing import Type, TypeVar

import mundi
import sidekick.api as sk
from .report_base import Report
from .. import fitting as fit
from ..models import SEAIR, Model
from ..utils import extract_keys

T = TypeVar("T")
INIT_KEYS = []


class SingleReport(Report):
    """
    A report based on a single run of a model.
    """

    model_cls: Type[Model] = SEAIR
    model: Model
    columns: list = ("date", "cases", "deaths")

    @classmethod
    def from_region(cls, region, *, model_cls=None, init_cases=True, **kwargs):
        """
        Initialize report from cases reported in region.
        """
        region = mundi.region(region)
        init_kwargs = extract_keys(INIT_KEYS, kwargs)
        model = (model_cls or cls.model_cls)(region=region, **kwargs)
        if init_cases:
            model.set_cases()
        return cls(model, **init_kwargs)

    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self._has_init_cases = False

    def init_cases(self, data=None, regions=None, raises=True, **kwargs):
        """
        Initialize model with statistics about cases.

        It can either receive an explicit data frame with cases/deaths
        statistics or a callable object that receives a model and return
        the desired cases.

        If none of these are passed, it assumes that the cases should be
        initialized from the region.
        """

        kwargs.setdefault("real", True)
        regions = {} if regions is None else regions
        region = self.model.region

        if data is None:
            if region is None:
                if not raises:
                    return self
                msg = "no cases data and model is not associated with a region."
                raise ValueError(msg)
            try:
                curves = regions[region]
            except KeyError:
                curves = regions[region] = region.pydemic.epidemic_curve(**kwargs)
        elif callable(data):
            curves = data(self.model)
        else:
            curves = data

        if len(curves) != 0:
            self.model.set_cases(curves)
        else:
            self.model.infect(1)

        self._has_init_cases = True
        return self

    def init_R0(self: T, method="RollingOLS", range=None) -> T:
        """
        Initialize R0 from cases data.
        """
        clinical_model = self.model.clinical()
        curves = clinical_model[["cases:dates", "deaths:dates"]]
        R0 = fit.estimate_R0(clinical_model, curves, Re=True, method=method)
        if R0.is_finite:
            if range:
                a, b = range
                self.model.R0 = max(a, min(b, R0.value))
            else:
                self.model.R0 = R0.value
        return self


INIT_KEYS.extend(sk.signature(SingleReport.__init__).parameters)
