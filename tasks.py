from invoke import task


@task
def test(ctx, all=False):
    if not all:
        ctx.run('pytest --maxfail=2 --lf -m "not slow"', pty=True)
    if all:
        ctx.run("pytest --cov", pty=True)
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
    test(ctx, all=True)
