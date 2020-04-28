from abc import ABC
from typing import Callable

import sidekick as sk

from ..models import Model, ODEModel
from ..utils import param_property, param_alias


class ClinicalModel(Model, ABC):
    """
    Base class for clinical models that track the infection curve and
    models the clinical history of patients.
    """
    CLINICAL_COMPONENTS = ("hospitalizations", "hospitalized", "deaths")
    DATA_ALIASES = {
        'H': 'hospitalized',
        'D': 'deaths',
    }

    # Delegates (population parameters)
    initial_population = sk.delegate_to('infection_model')

    # Properties and aliases
    case_fatality_rate = param_property()
    CFR = param_alias("case_fatality_rate")

    @property
    def empirical_CFR(self):
        return self["empirical_CFR"].iloc[-1]

    @property
    def empirical_IFR(self):
        return self["empirical_IFR"].iloc[-1]

    def __init__(self, infection_model, *args, **kwargs):
        self.infection_model = infection_model
        super().__init__(*args, **kwargs)

    #
    # Data accessors
    #
    def get_data(self, name):
        name = self.DATA_ALIASES.get(name, name)
        try:
            return super().get_data(name)
        except ValueError as e:
            return self.infection_model.get_data(name)

    # Basic columns
    def get_data_infectious(self):
        return self.infection_model["infectious"]

    def get_data_infected(self):
        try:
            return self.infection_model["infected"]
        except KeyError:
            return self.infection_model["cases"]

    def get_data_cases(self):
        return self.infection_model["cases"]

    # Derived methods
    def get_data_empirical_CFR(self):
        return (self["deaths"] / self["cases"]).fillna(0.0)

    def get_data_empirical_IFR(self):
        return (self["deaths"] / self["infected"]).fillna(0.0)

    # Abstract interface
    def get_data_deaths(self):
        raise NotImplementedError('must be implemented in sub-classes')

    def get_data_hospitalizations(self):
        raise NotImplementedError('must be implemented in sub-classes')

    def get_data_hospitalized(self):
        raise NotImplementedError('must be implemented in sub-classes')

    #
    # Other functions
    #
    def plot(self, components=None, *, show=False, **kwargs):
        if components is None:
            self.infection_model.plot(**kwargs)
            components = self.CLINICAL_COMPONENTS
        super().plot(components, show=show, **kwargs)


class ClinicalObserverModel(ClinicalModel, ABC):
    """
    A clinical model that derives all its dynamic variables from simple
    transformations from the infected population.

    Observer models do not control directly their own dynamics and simply
    delegate time evolution to the corresponding infectious model.
    """

    DATA_COLUMNS = ()

    # Delegates (times and dates)
    times = sk.delegate_to('infection_model')
    dates = sk.delegate_to('infection_model')
    iter = sk.delegate_to('infection_model')
    time = sk.delegate_to('infection_model')
    date = sk.delegate_to('infection_model')

    def __init__(self, model, params=None, *, date=None, **kwargs):
        if not (date is None or date == model.date):
            raise ValueError('cannot set date')
        super().__init__(model, params, date=model.date, **kwargs)

    def run_to_fill(self, data, times):
        raise RuntimeError

    def run(self, time):
        return self.infection_model.run(time)

    def run_until(self, condition: Callable[[Model], bool]):
        return self.infection_model.run_until(condition)


class ClinicalODEModel(ClinicalModel, ODEModel, ABC):
    """
    Base class for clinical models based on ordinary differential equations.
    """
