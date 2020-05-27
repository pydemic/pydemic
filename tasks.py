from invoke import task


@task
def test(ctx, all=False, report_xml=False, verbose=False, clear_cache=False):
    if clear_cache:
        ctx.run("rm -rf .pytest_cache")
    suffix = " -vv " if verbose else ""
    if not all:
        ctx.run(f'pytest --maxfail=2 --lf -m "not slow" {suffix}', pty=True)
    if all:
        suffix += " --cov-report xml" if report_xml else ""
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
    ctx.run("rm coverage.xml -rf")
    ctx.run("./cc-test-reporter before-build", pty=True)
    test(ctx, all=True, report_xml=True)
    ctx.run("echo 'COVERAGE.XML stats' && wc coverage.xml")
    ctx.run(
        "./cc-test-reporter after-build -r "
        "7e76aeb0890556339111bab973b18d3678572fdb2c63ea3bb388f3933066756b"
    )
