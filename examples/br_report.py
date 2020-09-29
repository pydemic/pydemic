"""
This script creates a report cases, deaths, and hospital pressure for Brazilian
municipalities or states
"""
from functools import lru_cache
from itertools import islice
from typing import Iterable

import click
import mundi
from mundi import Region

import sidekick.api as sk
from pydemic.models import SEAIR
from pydemic.report import GroupReport

REGION_QUERIES = {
    "state": {"type": "state"},
    "city": {"type": "city"},
    "macro": {"type": "region", "subtype": "macro_region"},
}


def regions(kind, truncate=None, skip=0, parent=None) -> Iterable[Region]:
    """
    Return a list of regions for the given kind.
    """
    try:
        query = REGION_QUERIES[kind]
    except KeyError:
        raise ValueError(f"Invalid region type: {kind}")

    out = mundi.regions(country_code="BR", **query).index
    if parent is not None:
        out = filter(is_ancestor(parent), map(mundi.region, out))

    if skip != 0:
        out = islice(out, skip, None)
    if truncate is not None:
        out = islice(out, truncate)
    return map(mundi.region, out)


@sk.curry(2)
@lru_cache(5000)
def is_ancestor(parent_id, reg):
    parent = reg.parent
    if parent is None:
        return False
    elif parent.id == parent_id:
        return True
    else:
        return is_ancestor(parent_id, parent)


@sk.curry(1)
def report(regions, R0_range=(0.5, 1.5), run=60, raises=False):
    print("[info] Starting models")
    return (
        GroupReport.from_options(SEAIR, region=regions)
        .log("Init cases")
        .init_cases(raises=raises)
        .log("Init R0")
        .init_R0(range=R0_range, raises=raises)
        .log("Running simulation")
        .run(run, raises=raises)
    )


@sk.curry(2)
def export(path: str, report: GroupReport, dtype=None, times=None):
    kwargs = {"columns": ["cases", "deaths", "severe", "critical", suspect], "dtype": dtype}
    if times is not None:
        kwargs["times"] = [int(x) for x in times.split(",")]
    info = ["region.sus_macro_id", "region.sus_macro_name"]
    data = report.report_time_columns_data(**kwargs, info=info)

    if path is None:
        print(data)
    else:
        data.to_pickle(path + ".pkl")
        data.to_csv(path + ".csv")
        data.to_excel(path)


def suspect(m, dates):
    return m["cases:dates"].loc[dates] * 0.10 / 0.13


@click.command()
@click.option("--kind", "-k", default="state", help="Select regions.")
@click.option("--debug", "-d", is_flag=True, help="Enable debugging tracebacks.")
@click.option("--out", "-o", help="Output file.")
@click.option("--float", "-f", help="Output data as floating point values.")
@click.option("--truncate", "-t", type=int, help="Limit the number of models.")
@click.option("--times", help="Comma separated list of times to consider in the output.")
@click.option("--skip", "-s", type=int, default=0, help="Skip that many models.")
@click.option("--parent", "-p", help="Parent element.")
@click.option("--strict", is_flag=True, help="Parent element.")
def main(kind, debug, out, float, truncate, skip, parent, times, strict):
    dtype = None if float else int
    try:
        sk.pipe(
            regions(kind, truncate=truncate, skip=skip, parent=parent),
            report,
            export(out, dtype=dtype, times=times),
        )
    except (ValueError, SystemExit) as ex:
        if debug:
            raise
        raise SystemExit(ex)


if __name__ == "__main__":
    main()
