from .. import formulas
from ..types import ComputedDict
from pydemic.types import inverse, transform, alias


class EpidemicParams(ComputedDict):
    """
    Common epidemiological parameters
    """

    __slots__ = ()
    R0: float = 1.0
    serial_period: float = 1.0
    K: float = transform("(R0 - 1) / serial_period", "K * serial_period + 1")
    duplication_time: float = transform("log(2) / K", "log(2) / tau")


class SIRParams(EpidemicParams):
    """
    Declare parameters for the SIR family of models.
    """

    __slots__ = ()

    infectious_period: float = inverse("gamma")
    serial_period: float = alias("infectious_period")
    gamma: float = 1.0

    # Derived parameters and expressions
    beta: float = transform(formulas.sir.beta.formula)
    K: float = transform(formulas.sir.K.formula, formulas.sir.R0_from_K.formula)


class SEIRParams(SIRParams):
    """
    Declare parameters for the SEIR model, extending SIR.
    """

    __slots__ = ()
    beta: float = transform(formulas.seir.beta.formula)
    K: float = transform(formulas.seir.beta.formula, formulas.seir.R0_from_K)
    serial_period: float = transform("infectious_period + incubation_period")

    # We recover SIR from SEIR by setting sigma -> infinity.
    # We instead assign a very small value to sigma ZeroDivision errors
    incubation_period: float = 1.0
    sigma: float = inverse("incubation_period")


class SEAIRParams(SEIRParams):
    """
    Declare parameters for the SEAIR model, extending SEIR.
    """

    __slots__ = ()
    beta: float = transform(formulas.seair.beta.formula)
    K: float = transform(formulas.seair.beta.formula, formulas.seair.R0_from_K)

    # SEIR assumes all cases are symptomatic. This disables the "asymptomatic"
    # compartment.
    prob_symptoms: float = 1.0
    Qs: float = alias("prob_symptoms")

    # The rationale for Rho defaulting to 1 is that the difference between
    # regular and asymptomatic cases is simply a matter of notification.
    # This makes asymptomatic equivalent to symptomatic from an
    # epidemiological point of view.
    rho: float = 1.0


class ClinicalParams(ComputedDict):
    """
    Basic clinical parameters of a disease.
    """

    # Basic mortality statistics
    case_fatality_ratio: float = 1.0
    CFR: float = alias("case_fatality_ratio")

    infection_fatality_ratio: float = 1.0
    IFR: float = alias("infection_fatality_ratio")

    prob_critical: float = 0.0
    Qcr: float = alias("prob_critical")

    prob_severe: float = 0.0
    Qsv: float = alias("prob_severe")

    # Derived statistics
    hospital_fatality_ratio: float = transform("case_fatality_ratio / prob_severe")
    HFR: float = alias("hospital_fatality_ratio")

    icu_fatality_ratio: float = transform("case_fatality_ratio / prob_critical")
    ICUFR: float = alias("icu_fatality_ratio")

    prob_aggravate_to_icu: float = transform("prob_critical * prob_severe")

    # Delays and periods
    critical_period: float = 0.0
    severe_period: float = 0.0
    severe_delay: float = 0.0
    critical_delay: float = 0.0

    # Health system capacity and efficiency
    hospitalization_overflow_bias: float = 0.0
    icu_capacity: float = float("inf")
    hospital_capacity: float = float("inf")
    icu_occupancy: float = 0.75
    hospital_occupancy: float = 0.75
    icu_surge_capacity = transform("icu_capacity * (1 - icu_occupancy)")
    hospital_surge_capacity = transform("hospital_capacity * (1 - hospital_occupancy)")
