import pandas as pd
import requests
import sidekick as sk

import mundi
from mundi import transforms
from ..cache import tle_cache
from ..logging import log

HOURS = 3600
EPIDEMIC_CURVES_APIS = {}


def register_api(key):
    return lambda fn: EPIDEMIC_CURVES_APIS.setdefault(key, fn)


def epidemic_curve(region, api="auto", extra=False, **kwargs):
    """
    Universal interface to all epidemic curve loaders.

    Always return a dataframe with ["cases", "deaths"] columns for the given
    region. Some API's may offer additional columns such as "recovered", "test"
    etc.
    """
    code = mundi.code(region)
    fn = EPIDEMIC_CURVES_APIS[api]
    data = fn(code, **kwargs)
    return data if extra else data[["cases", "deaths"]]


@register_api("auto")
def auto_api(code, **kwargs):
    """
    Select best API to load according to region code.
    """
    if code == "BR" or code.startswith("BR-"):
        return brasil_io(code)
    elif len(code) == 2:
        return corona_api(code, **kwargs)
    raise ValueError(f"no API can load region with code: {code}")


@register_api("corona-api.com")
@tle_cache("covid-19", timeout=6 * HOURS)
@sk.retry(10, sleep=0.5)
def corona_api(code) -> pd.DataFrame:
    """
    Load country's cases, deaths and recovered timeline from corona-api.com.
    """
    url = "http://corona-api.com/countries/{code}?include=timeline"
    response = requests.get(url.format(code=code))
    data = response.json()
    df = pd.DataFrame(data["data"]["timeline"]).rename({"confirmed": "cases"}, axis=1)
    df = df[["date", "cases", "deaths", "recovered"]]
    df.index = pd.to_datetime(df.pop("date"))
    df = df[df.fillna(0).sum(1) > 0]
    return df.sort_index()


@register_api("brasil.io")
def brasil_io(code):
    cases = brasil_io_cases()
    cases = cases[cases["id"] == code].drop(columns="id")
    cases = cases.drop_duplicates("date")
    return cases.set_index("date").sort_index()


@tle_cache("covid-19", timeout=12 * HOURS)
@sk.retry(10, sleep=0.5)
def brasil_io_df() -> pd.DataFrame:
    url = "https://data.brasil.io/dataset/covid19/caso_full.csv.gz"
    return pd.read_csv(url)


# @tle_cache("covid-19", timeout=12 * HOURS)
@sk.lru_cache(1)
def brasil_io_cases():
    df = brasil_io_df()
    cols = {
        "last_available_confirmed": "cases",
        "confirmed": "cases",
        "last_available_deaths": "deaths",
        "city_ibge_code": "code",
    }

    cases = df.rename(cols, axis=1)
    cases = cases[cases["code"].notna()]

    cases["code"] = cases["code"].apply(lambda x: str(int(x))).astype("string")
    cases["code"] = "BR-" + cases["code"]

    cases["date"] = pd.to_datetime(cases["date"])
    cases = cases[cases["place_type"] == "city"]

    cases = cases[["date", "code", "cases", "deaths"]]
    cases = cases.dropna().reset_index(drop=True)
    cases = cases.rename({"code": "id"}, axis=1)

    log.info("Merging data from brasil.io")

    result = {}
    for col in ["cases", "deaths"]:
        data = cases.pivot_table(index="id", columns="date", values=col).fillna(-1).sort_index()
        data = transforms.sum_children(data).reset_index()
        data = data.melt(id_vars=["id"], var_name="date", value_name=col)
        data = data[data[col] >= 0]
        result[col] = data
    return (
        pd.merge(*result.values(), on=["id", "date"], how="outer")
        .fillna(0)
        .astype({"cases": int, "deaths": int})
    )


if __name__ == "__main__":
    sk.import_later("..cli.api:covid19_api_downloader", package=__package__)()
