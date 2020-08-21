import numpy as np

import mundi
from pydemic.api.covid19 import download_brasil_io_cases
from pydemic_ui import st


@st.cache
def ibge_to_mundi_codes():
    data = {}
    for kind in ["city"]:  # , "state"]:
        children = mundi.regions(country_id="BR", type=kind).mundi["numeric_code"]
        data.update(children["numeric_code"].to_dict())
    return dict((int(v), k) for k, v in data.items())


#
# parent = region('BR')
# children = parent.children(which='primary')
#
# data = parent.pydemic.epidemic_curve()
# data_children = pd.concat(
#     [r.pydemic.epidemic_curve() for r in children],
#     keys=[r.id for r in children],
#     axis=1,
# )
#
# data.plot()
# st.pyplot()
#
# data_children.plot()
# st.pyplot()
#
# x1 = data_children.iloc[:, ::2].sum(1)
# x2 = data.iloc[:, [0]].sum(1)
# x3 = pd.concat([x1, x2], axis=1)
# st.write(x3)
#
# sp = region('BR-351561')
# x4 = sp.pydemic.epidemic_curve()
# st.text(sp.children())


@st.cache
def get_cases():
    return download_brasil_io_cases()


@st.cache
def get_curves():
    data = get_cases()
    cols = {
        "last_available_confirmed": "cases",
        "confirmed": "cases",
        "last_available_deaths": "deaths",
        "city_ibge_code": "id",
    }
    data = data.rename(columns=cols)[["date", "id", "cases", "deaths"]]
    data["id"] = data["id"].apply(ibge_to_mundi_codes().get).dropna()

    cases, deaths = (
        data.pivot_table(index="id", columns="date", values=col, fill_value=0).sort_index()
        for col in ["cases", "deaths"]
    )

    print(cases)

    return cases, deaths


mundi.Region
