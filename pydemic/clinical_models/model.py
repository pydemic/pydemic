from abc import ABC
from typing import Callable

import sidekick as sk

from ..models import Model, ODEModel


class ClinicalModel(Model, ABC):
    """
    Base class for clinical models that track the infection curve and
    models the clinical history of patients.
    """

    # Expose columns from the infectious model
    infectious = property(lambda self: self.infection_model["infectious"])
    cases = property(lambda self: self.infection_model["cases"])

    # Delegates (population parameters)
    initial_population = sk.delegate_to('infection_model')

    def __init__(self, infection_model, *args, **kwargs):
        self.infection_model = infection_model
        super().__init__(*args, **kwargs)

    def get_data(self, name):
        try:
            return super().get_data(name)
        except ValueError as e:
            return self.infection_model.get_data(name)


class ClinicalObserverModel(ClinicalModel, ABC):
    """
    A clinical model that derives all its dynamic variables from simple
    transformations from the infected population.

    Observer models do not control directly their own dynamics and simply
    delegate time evolution to the corresponding infectious model.
    """

    # Delegates (times and dates)
    times = sk.delegate_to('infection_model')
    dates = sk.delegate_to('infection_model')
    iter = sk.delegate_to('infection_model')
    time = sk.delegate_to('infection_model')
    date = sk.delegate_to('infection_model')

    def run_to_fill(self, data, times):
        pass

    def run(self, time):
        return self.infection_model.run(time)

    def run_until(self, condition: Callable[[Model], bool]):
        return self.infection_model.run_until(condition)


class ClinicalODEModel(ClinicalModel, ODEModel, ABC):
    """
    Base class for clinical models based on ordinary differential equations.
    """
