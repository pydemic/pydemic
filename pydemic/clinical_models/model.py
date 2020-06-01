from abc import ABC

import sidekick as sk

from ..models import Model, ODEModel
from ..utils import param_property, param_alias


class ClinicalModel(Model, ABC):
    """
    Base class for clinical models that track the infection curve and
    models the clinical history of patients.
    """

    class Meta:
        model_name = "Clinical"
        data_aliases = {"H": "hospitalized", "D": "deaths"}
        plot_columns = ("hospitalized_cases", "hospitalized", "deaths")

    # Delegates (population parameters)
    population = sk.delegate_to("infection_model")
    K = sk.delegate_to("infection_model")
    disease = sk.delegate_to("infection_model")
    disease_params = sk.delegate_to("infection_model")
    region = sk.delegate_to("infection_model")
    age_distribution = sk.delegate_to("infection_model")
    age_pyramid = sk.delegate_to("infection_model")

    # Properties and aliases
    case_fatality_ratio = param_property(default=0.0)
    infection_fatality_ratio = param_property(default=lambda _: _.CFR)
    CFR = param_alias("case_fatality_ratio")
    IFR = param_alias("infection_fatality_ratio")

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

        kwargs.setdefault("name", infection_model.name)
        super().__init__(*args, **kwargs)

    def __getattr__(self, item):
        try:
            return getattr(self.infection_model, item)
        except AttributeError:
            name = type(self).__name__
            raise AttributeError(f'"{name}" object has no "{item}" attribute')

    def copy(self, **kwargs):
        kwargs["infection_model"] = self.infection_model.copy()
        return super().copy(**kwargs)

    #
    # Data accessors
    #
    def get_column(self, name, idx):
        name = self.meta.data_aliases.get(name, name)
        try:
            return super().get_column(name, idx)
        except ValueError:
            return self.infection_model.get_column(name, idx)

    # Basic columns
    def get_data_population(self, idx):
        """
        Total population minus deaths.
        """
        return self.infection_model["population", idx] - self["deaths", idx]

    def get_data_infectious(self, idx):
        """
        Infectious population according to infectious model.

        This is usually the starting point of all clinical models.
        """
        return self.infection_model["infectious", idx]

    def get_data_cases(self, idx):
        """
        Cumulative curve of cases.

        A case is typically defined as an individual who got infected AND
        developed recognizable clinical symptoms.
        """
        return self.infection_model["cases", idx]

    def get_data_infected(self, idx):
        """
        Cumulative curve of infected individuals.

        Infected individuals might not develop clinical symptoms. They may never
        develop symptoms (asymptomatic) or develop them in a future time.
        """
        try:
            return self.infection_model["infected", idx]
        except KeyError:
            return self["cases", idx]

    # Derived methods
    def get_data_empirical_CFR(self, idx):
        """
        Empirical CFR computed as current deaths over cases.
        """
        return (self["deaths", idx] / self["cases", idx]).fillna(0.0)

    def get_data_empirical_IFR(self, idx):
        """
        Empirical IFR computed as current deaths over infected.
        """
        return (self["deaths", idx] / self["infected", idx]).fillna(0.0)

    # Abstract interface
    def get_data_death_rate(self, idx):
        """
        Daily number of deaths.
        """
        return self["deaths", idx].diff().fillna(0)

    def get_data_deaths(self, idx):
        """
        Cumulative curve of deaths.
        """
        raise NotImplementedError("must be implemented in sub-classes")

    def get_data_severe(self, idx):
        """
        Current number of severe cases.

        Severe cases usually require hospitalization, but have a low death risk.
        """
        raise NotImplementedError("must be implemented in sub-classes")

    def get_data_severe_cases(self, idx):
        """
        Cumulative number of severe cases.

        Severe cases usually require hospitalization, but have a low death risk.
        """
        raise NotImplementedError("must be implemented in sub-classes")

    def get_data_critical(self, idx):
        """
        Current number of critical cases.

        Critical cases require intensive care and are at a high risk of death.
        """
        raise NotImplementedError("must be implemented in sub-classes")

    def get_data_critical_cases(self, idx):
        """
        Cumulative number of critical cases.

        Critical cases require intensive care and are at a high risk of death.
        """
        raise NotImplementedError("must be implemented in sub-classes")

    def get_data_hospitalized(self, idx):
        """
        Cases currently occupying a hospital bed.

        In an ideal world, this would be equal to the number of severe cases.
        The default implementation assumes that.
        """
        return self["severe", idx]

    def get_data_hospitalized_cases(self, idx):
        """
        Cumulative number of hospitalizations.

        Default implementation assumes equal to the number of severe cases.
        """
        return self["severe_cases", idx]

    def get_data_icu(self, idx):
        """
        Number of ICU patients.

        In an ideal world, this would be equal to the number of critical cases.
        The default implementation assumes that.
        Default implementation assumes equal to the number of critical cases.
        """
        return self["critical", idx]

    def get_data_icu_cases(self, idx):
        """
        Cumulative number of ICU patients.

        Default implementation assumes equal to the number of critical cases.
        """
        return self["critical_cases", idx]

    #
    # Other functions
    #
    def plot(self, components=None, *, ax=None, logy=False, show=False, **kwargs):
        if components is None:
            self.infection_model.plot(**kwargs)
            components = self.meta.plot_columns
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

    # Methods delegates
    set_ic = sk.delegate_to("infection_model")
    set_data = sk.delegate_to("infection_model")
    set_cases = sk.delegate_to("infection_model")
    run = sk.delegate_to("infection_model")
    run_until = sk.delegate_to("infection_model")
    trim_dates = sk.delegate_to("infection_model")
    reset = sk.delegate_to("infection_model")
    epidemic_model_name = sk.delegate_to("infection_model")

    def __init__(self, model, params=None, *, date=None, **kwargs):
        if not (date is None or date == model.date):
            raise ValueError("cannot set date")
        super().__init__(model, params, date=model.date, **kwargs)

    def _initial_state(self):
        return ()

    def run_to_fill(self, data, times):
        raise RuntimeError


class ClinicalODEModel(ClinicalModel, ODEModel, ABC):
    """
    Base class for clinical models based on ordinary differential equations.
    """
