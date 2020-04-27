import warnings

from invoke import task


@task
def test(ctx):
    ctx.run("pytest --cov")
    ctx.run("black --check .")
    ctx.run("pycodestyle")
