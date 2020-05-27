import click
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


@click.command()
@click.argument("code")
@click.option("--api", "-a", default="auto", help="Force some specific API")
@click.option("--output", "-o", help="Save results to the given output")
@click.option("--plot", "-p", is_flag=True, help="Plot results")
@click.option("--log", "-l", is_flag=True, help="Use logarithmic scale")
@click.option("--daily", "-d", is_flag=True, help="Show new cases per day")
@click.option("--grid", "-g", is_flag=True, help="Add grid to the graph")
def covid19_api_downloader(code, api, output, plot, log, daily, grid):
    """
    Load data of COVID-19 cases and deaths.
    """
    from ..api.covid19 import epidemic_curve

    df = epidemic_curve(code, api)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Daily cases?
    if daily:
        data = np.diff(df.values, prepend=0, axis=0)
        df = pd.DataFrame(data, columns=df.columns, index=df.index)

    # Plotting
    if plot:
        df.plot.bar(logy=log, width=0.8) if daily else df.plot(logy=log)
        if grid:
            plt.grid()
        plt.title(code)
        plt.show()

    # Save output file
    def exts(lst):
        for ext in lst:
            if output.endswith("." + ext):
                return True
        return False

    if not output:
        if not plot:
            print(df)
    elif exts(["csv", "csv.gz"]):
        df.to_csv(output)
    elif exts(["csv", "csv.gz"]):
        df.to_pickle(output)
    elif exts(["xls", "xlsx"]):
        df.to_excel(output)
    else:
        raise SystemExit(f"Invalid output file: {output!r}")
