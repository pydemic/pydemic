import pandas as pd
import requests
import sidekick as sk

import mundi
from ..cache import tle_cache

HOURS = 3600
EPIDEMIC_CURVES_APIS = {}


def register_api(key):
    return lambda fn: EPIDEMIC_CURVES_APIS.setdefault(key, fn)


def epidemic_curve(region, api="auto", extra=False, **kwargs):
    """
    Universal interface to all epidemic curve loaders.

    Always return a dataframe with ["cases", "fatalities"] columns for the given
    region. Some API's may offer additional columns such as "recovered", "test"
    etc.
    """
    code = mundi.code(region)
    fn = EPIDEMIC_CURVES_APIS[api]
    data = fn(code, **kwargs)
    return data if extra else data[["cases", "fatalities"]]


@register_api("auto")
def auto_api(code, **kwargs):
    """
    Select best API to load according to region code.
    """
    if len(code) == 2:
        return corona_api(code, **kwargs)
    elif code.startswith("BR-"):
        return brasil_io(code)
    else:
        raise ValueError(f"no API can load region with code: {code}")


@register_api("corona-api.com")
@tle_cache("covid-19", timeout=6 * HOURS)
@sk.retry(10, sleep=0.5)
def corona_api(code) -> pd.DataFrame:
    """
    Load country's cases, fatalities and recovered timeline from corona-api.com.
    """
    url = "http://corona-api.com/countries/{code}?include=timeline"
    response = requests.get(url.format(code=code))
    data = response.json()
    df = pd.DataFrame(data["data"]["timeline"]).rename(
        {"confirmed": "cases", "deaths": "fatalities"}, axis=1
    )
    df = df[["date", "cases", "fatalities", "recovered"]]
    df.index = pd.to_datetime(df.pop("date"))
    df = df[df.fillna(0).sum(1) > 0]
    return df.sort_index()


@register_api("brasil.io")
def brasil_io(code):
    cases = brasil_io_cases()
    cases = cases[cases["code"] == code].drop(columns="code")
    cases = cases.drop_duplicates("date")
    return cases.set_index("date").sort_index()


@tle_cache("covid-19", timeout=12 * HOURS)
@sk.retry(10, sleep=0.5)
def brasil_io_df() -> pd.DataFrame:
    url = "https://data.brasil.io/dataset/covid19/caso_full.csv.gz"
    return pd.read_csv(url)


def brasil_io_cases():
    df = brasil_io_df()
    cols = {
        "last_available_confirmed": "cases",
        "confirmed": "cases",
        "last_available_deaths": "fatalities",
        "deaths": "fatalities",
        "city_ibge_code": "code",
    }

    cases = df.rename(cols, axis=1)
    cases = cases[cases["code"].notna()]

    cases["code"] = cases["code"].apply(lambda x: str(int(x))).astype("string")
    cases.loc[cases["place_type"] == "state", "code"] = cases["state"]
    cases["code"] = "BR-" + cases["code"]

    cases["date"] = pd.to_datetime(cases["date"])

    cases = cases[["date", "code", "cases", "fatalities"]]
    return cases.dropna().reset_index(drop=True)


if __name__ == "__main__":
    sk.import_later("..cli.api:covid19_api_downloader", package=__package__)()
