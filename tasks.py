from invoke import task


@task
def test(ctx, all=False):
    ctx.run('pytest --maxfail=2 --lf -m "not slow"', pty=True)
    if all:
        ctx.run("pytest --cov", pty=True)
        ctx.run("black --check .")
        ctx.run("pycodestyle")


@task
def cov(ctx, report=True):
    ctx.run("pytest --cov --cov-report=html", pty=True)
