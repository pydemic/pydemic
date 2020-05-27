import time
from functools import lru_cache

import pandas as pd
import requests
import sidekick as sk

import mundi
from mundi import transforms
from ..cache import ttl_cache
from ..logging import log
from ..utils import today

HOURS = 3600
TIMEOUT = 6 * HOURS
EPIDEMIC_CURVES_APIS = {}
MOBILITY_DATA_APIS = {}


def epidemic_curve_api(key):
    return lambda fn: EPIDEMIC_CURVES_APIS.setdefault(key, fn)


def mobility_data_api(key):
    return lambda fn: MOBILITY_DATA_APIS.setdefault(key, fn)


#
# Epidemic curves
#
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


@epidemic_curve_api("auto")
def auto_api(code, **kwargs):
    """
    Select best API to load according to region code.
    """
    if code == "BR" or code.startswith("BR-"):
        return brasil_io(code)
    elif len(code) == 2:
        return corona_api(code, **kwargs)
    raise ValueError(f"no API can load region with code: {code}")


@epidemic_curve_api("corona-api.com")
@ttl_cache("covid-19", timeout=TIMEOUT)
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


@epidemic_curve_api("brasil.io")
def brasil_io(code):
    cases = brasil_io_cases()
    cases = cases[cases["id"] == code].drop(columns="id")
    cases = cases.drop_duplicates("date")
    return cases.set_index("date").sort_index()


@ttl_cache("covid-19", timeout=TIMEOUT)
@sk.retry(10, sleep=0.5)
def brasil_io_cases() -> pd.DataFrame:
    """
    Return the complete dataframe of cases and deaths from Brasil.io.
    """
    url = "https://data.brasil.io/dataset/covid19/caso_full.csv.gz"
    df = pd.read_csv(url)
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


#
# Mobility data
#
@ttl_cache("covid-19", timeout=TIMEOUT)
@sk.retry(10, sleep=0.5)
def google_mobility_data(cli=False):
    """
    Download google mobility data
    """
    url = "https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv"

    log.info(f"Downloading google mobility data {today()}")
    t0 = time.time()
    data = requests.get(url)
    log.info(f"Download finished after {time.time() - t0:0.2} seconds")

    data_cols = ["retail", "grocery", "parks", "transit", "work", "residential"]

    df = pd.read_csv(data.content.decode("utf8")).rename(
        {
            "retail_and_recreation_percent_change_from_baseline": "retail",
            "grocery_and_pharmacy_percent_change_from_baseline": "grocery",
            "parks_percent_change_from_baseline": "parks",
            "transit_stations_percent_change_from_baseline": "transit",
            "workplaces_percent_change_from_baseline": "work",
            "residential_percent_change_from_baseline": "residential",
        },
        axis=1,
    )
    df["date"] = pd.to_datetime(df["date"])
    df[data_cols] = df[data_cols] / 100.0
    return df


def fix_google_mobility_data_region_codes(df):
    print(df)
    data = df[["country_region_code", "sub_region_1", "sub_region_2"]]
    codes = data.apply(subregion_code)
    print(codes)
    return df


@lru_cache(1024)
def subregion_code(country, region, subregion):
    region = region or None
    subregion = subregion or None

    # Check arbitrary mapping
    mapping = google_mobility_map_codes()
    try:
        return mapping[country, region, subregion]
    except KeyError:
        pass

    # Fasttrack pure-country codes
    if not region:
        return country

    for name in (subregion, region):
        try:
            region = mundi.region(country_code=country, name=name)
        except LookupError:
            return region.id

    raise ValueError(country, region, subregion)
    return country + "-" + region


@lru_cache(1)
def google_mobility_map_codes() -> dict:
    data = {}

    # Brazilian states
    for state in mundi.regions("BR", type="state"):
        data["BR", f"State of {state}", None] = state.id
    data["BR", "Federal District", None] = "BR-DF"

    return data


if __name__ == "__main__":
    # sk.import_later("..cli.api:covid19_api_downloader", package=__package__)()

    df = google_mobility_data()
    df = fix_google_mobility_data_region_codes(df)
    print(df)
