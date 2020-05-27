from invoke import task


@task
def test(ctx, all=False, report_xml=False, verbose=False):
    suffix = " -vv " if verbose else ""
    if not all:
        ctx.run(f'pytest --maxfail=2 --lf -m "not slow" {suffix}', pty=True)
    if all:
        suffix += " --cov-report=xml" if report_xml else ""
        ctx.run(f"pytest --cov {suffix}", pty=True)
        style(ctx)


@task
def style(ctx):
    ctx.run("black --check .")
    ctx.run("flake8 pydemic")


@task
def cov(ctx, report=True):
    ctx.run("pytest --cov --cov-report=html --cov-report=term", pty=True)


@task
def ci(ctx):
    """
    Non-configurable task that is run in continuous integration.
    """
    test(ctx, all=True, report_xml=True)
