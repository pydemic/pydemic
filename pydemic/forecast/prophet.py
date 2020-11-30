import pandas as pd
from fbprophet import Prophet
from matplotlib import pyplot as plt
from mundi import region

from ..region import RegionT
from pydemic_ui import st

plt.gcf()

for r in ["US", "DE", "IT", "CN", "ES", "BR", "BR-SP", "BR-MG", "BR-PA", "BR-MS"]:
    br: RegionT = region(r)
    st.header(br.name)

    cases = br.pydemic.epidemic_curve()
    new = cases.diff().iloc[15:]
    new.plot(grid=True, logy=True)
    st.pyplot()

    col: pd.Series = new["cases"]
    # col = trim_zeros(col.where(col < 10, 0))

    Y = col.reset_index().rename(columns={"date": "ds", "cases": "y"})
    Y["cap"] = cap = 100 * Y["y"].max()

    m = Prophet(
        changepoint_range=0.9,
        weekly_seasonality=True,
        changepoint_prior_scale=0.25,
        seasonality_mode="multiplicative",
        growth="logistic",
    )
    m.fit(Y)

    future = m.make_future_dataframe(60)
    future["cap"] = cap
    pred = m.predict(future)
    m.plot(pred, plot_cap=False)
    st.pyplot()

    m.plot_components(pred, plot_cap=False)
    st.pyplot()
