from abc import ABC

import click
from .report import Report


class ReportCLI(ABC):
    """
    Abstract class for command line interfaces.
    """

    report_class: "Report"

    def run(self):
        decorated = self.prepare(self.main)
        return decorated()

    def main(self, **kwargs):
        raise NotImplementedError

    def prepare(self, main):
        @click.command()
        @click.argument("region")
        @click.option("--R0", type=float, help="Basic reproductive number")
        @click.option("--output", "-o", default="report.xlsx", type=click.Path(), help="Output")
        @click.option("--columns", "-c", help="Display columns")
        def func(**kwargs):
            return main(**kwargs)

        return func


class SingleReportCLI(ReportCLI):
    """
    CLI for single report sub-classes.
    """

    def main(self, region, r0, output, columns, **kwargs):
        if r0 is not None:
            kwargs["R0"] = r0
        if columns is not None:
            columns = [c.strip() for c in columns.split(",")]
            kwargs["columns"] = columns

        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        report = self.report_class.from_region(region, **kwargs)
        click.echo("Running simulation...")
        report.analyze()
        click.echo("Done!")

        if output == "stdout":
            report.show()
        else:
            report.save(output)
            click.echo(f"Results saved to {output}")


class MultiReportCLI(ReportCLI):
    """
    CLI for single report sub-classes.
    """
