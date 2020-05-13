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
    DATA_ALIASES = {"H": "hospitalized", "D": "deaths"}
    model_name = "Clinical"

    # Delegates (population parameters)
    population = sk.delegate_to("infection_model")
    K = sk.delegate_to("infection_model")
    disease = sk.delegate_to("infection_model")

    # Properties and aliases
    case_fatality_rate = param_property(default=0.0)
    infection_fatality_rate = param_property(default=lambda _: _.CFR)
    CFR = param_alias("case_fatality_rate")
    IFR = param_alias("infection_fatality_rate")

    @property
    def empirical_CFR(self):
        return self["empirical_CFR:final"]

    @property
    def empirical_IFR(self):
        return self["empirical_IFR:final"]

    @property
    def clinical_model(self):
        return self

    def __init__(self, infection_model, *args, **kwargs):
        self.infection_model = infection_model

        for k in ("disease", "region"):
            if k not in kwargs:
                kwargs[k] = getattr(infection_model, k)

        super().__init__(*args, **kwargs)

    def __getattr__(self, item):
        try:
            return getattr(self.infection_model, item)
        except AttributeError:
            name = type(self).__name__
            raise AttributeError(f'"{name}" object has no "{item}" attribute')

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
    def get_data_population(self):
        """
        Total population minus deaths.
        """
        return self.infection_model["population"] - self["deaths"]

    def get_data_infectious(self):
        """
        Infectious population according to infectious model.

        This is usually the starting point of all clinical models.
        """
        return self.infection_model["infectious"]

    def get_data_cases(self):
        """
        Cumulative curve of cases.

        A case is typically defined as an individual who got infected AND
        developed recognizable clinical symptoms.
        """
        return self.infection_model["cases"]

    def get_data_infected(self):
        """
        Cumulative curve of infected individuals.

        Infected individuals might not develop clinical symptoms. They may never
        develop symptoms (asymptomatic) or develop them in a future time.
        """
        try:
            return self.infection_model["infected"]
        except KeyError:
            return self["cases"]

    # Derived methods
    def get_data_empirical_CFR(self):
        """
        Empirical CFR computed as current deaths over cases.
        """
        return (self["deaths"] / self["cases"]).fillna(0.0)

    def get_data_empirical_IFR(self):
        """
        Empirical IFR computed as current deaths over infected.
        """
        return (self["deaths"] / self["infected"]).fillna(0.0)

    # Abstract interface
    def get_data_death_rate(self):
        """
        Daily number of deaths.
        """
        return self["deaths"].diff().fillna(0)

    def get_data_deaths(self):
        """
        Cumulative curve of deaths.
        """
        raise NotImplementedError("must be implemented in sub-classes")

    def get_data_severe(self):
        """
        Current number of severe cases.

        Severe cases usually require hospitalization, but have a low death risk.
        """
        raise NotImplementedError("must be implemented in sub-classes")

    def get_data_severe_cases(self):
        """
        Cumulative number of severe cases.

        Severe cases usually require hospitalization, but have a low death risk.
        """
        raise NotImplementedError("must be implemented in sub-classes")

    def get_data_critical(self):
        """
        Current number of critical cases.

        Critical cases require intensive care and are at a high risk of death.
        """
        raise NotImplementedError("must be implemented in sub-classes")

    def get_data_critical_cases(self):
        """
        Cumulative number of critical cases.

        Critical cases require intensive care and are at a high risk of death.
        """
        raise NotImplementedError("must be implemented in sub-classes")

    def get_data_hospitalized(self):
        """
        Cases currently occupying a hospital bed.

        In an ideal world, this would be equal to the number of severe cases.
        The default implementation assumes that.
        """
        return self["severe"]

    def get_data_hospitalized_cases(self):
        """
        Cumulative number of hospitalizations.

        Default implementation assumes equal to the number of severe cases.
        """
        return self["severe_cases"]

    def get_data_icu(self):
        """
        Number of ICU patients.

        In an ideal world, this would be equal to the number of critical cases.
        The default implementation assumes that.
        Default implementation assumes equal to the number of critical cases.
        """
        return self["critical"]

    def get_data_icu_cases(self):
        """
        Cumulative number of ICU patients.

        Default implementation assumes equal to the number of critical cases.
        """
        return self["critical_cases"]

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
    times = sk.delegate_to("infection_model")
    dates = sk.delegate_to("infection_model")
    iter = sk.delegate_to("infection_model")
    time = sk.delegate_to("infection_model")
    date = sk.delegate_to("infection_model")
    state = sk.delegate_to("infection_model")

    def __init__(self, model, params=None, *, date=None, **kwargs):
        if not (date is None or date == model.date):
            raise ValueError("cannot set date")
        super().__init__(model, params, date=model.date, **kwargs)

    def _initial_state(self):
        return ()

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
