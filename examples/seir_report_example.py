import pandas as pd
from matplotlib import pyplot as plt

import mundi
from pydemic.diseases import covid19
from pydemic.models import SEAIR
from pydemic.plot import color, mark_x, mark_y

macros = mundi.regions_dataframe(parent_id="BR-MG", subtype="healthcare region")
R0 = {
    "BR-MG": 1.27,  # Minas Gerais
    "BR-3106200": 1.09,  # Belo Horizonte
    "BR-3106705": 1.19,  # Betim
    "BR-3136702": 1.14,  # Juiz de Fora
    "BR-3167202": 1.10,  # Sete Lagoas
    **{k: 1.27 for k in macros.index},
}


# Initialize model
def get_model(region):
    region = mundi.region(region)
    data = region.pydemic.epidemic_curve()
    empirical_CFR = (data["deaths"] / data["cases"]).mean()
    notification_rate = min(0.5, covid19.CFR(region=region) / empirical_CFR)

    m = SEAIR(region=region, R0=R0[region.id])
    real_data = data.copy()
    real_data["cases"] /= notification_rate
    m.set_cases(real_data, save_observed=True)
    m.run(60)
    return m.clinical.overflow_model()


def plot_curves(model, data=None):
    if data is None:
        data = model.info["observed.curves"]

    model["cases:dates"].plot(label="Simulated cases")
    data["cases"].plot(label="Observed cases", color=color(), lw=2, ls="--")

    model["deaths:dates"].plot(label="Simulated deaths")
    data["deaths"].plot(label="Observed deaths", color=color(), lw=2, ls="--")
    mark_x(model.info["event.simulation_start"].date, "k:")

    plt.grid("on")
    plt.yscale("log")
    plt.ylim(1, None)
    plt.legend()
    plt.title(model.region.name)


def plot_hospitalization_curves(model):
    data = model[["critical", "severe"]].rename(columns={"critical": "UTI", "severe": "Clínico"})
    data.index = model.dates
    data.plot(logy=True)

    mark_y(model.region.icu_capacity, "k--", text="UTI")
    mark_y(model.region.hospital_capacity, "--", color="0.5", text="Leitos clínicos")
    mark_x(model.info["event.simulation_start"].date, "k:")
    plt.title(model.region.name)
    plt.grid("on")


def save_data(model):
    data = model[["cases", "deaths", "severe", "critical", "ppe"]]
    data.index = model.dates
    data = data.iloc[model.info["event.simulation_start"].time :]
    data["name"] = model.region.name
    data["population"] = model.region.population
    data["R0"] = model.R0
    data.to_excel(f"results/data-{model.region.id}.xlsx")


for ref in R0:
    print(f"Processing {ref}...")

    m = get_model(ref)
    plot_curves(m)
    plt.savefig(f"results/cases-{m.region.id}.svg")
    plt.clf()

    plot_hospitalization_curves(m)
    plt.savefig(f"results/hospitalization-{m.region.id}.svg")
    plt.clf()

    save_data(m)
