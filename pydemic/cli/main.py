from pprint import pformat


def main(cls, *args, **kwargs):
    """
    Executes the default action for the model. Convenient for making quick
    and dirt CLI tools.
    """
    import click

    kind_map = {"int": int, "float": float, "str": str}

    @click.option("--plot", is_flag=True, help="Display plot")
    @click.option("--debug", is_flag=True, help="Display debug information")
    def cli(plot=False, debug=False, **kwargs_):
        kwargs_ = {k: v for k, v in kwargs_.items() if v is not None}
        kwargs_ = {**kwargs, **kwargs_}
        m = cls._main(*args, **kwargs_)
        m.run()
        print(m)
        if debug:
            print("\n\nDEBUG SYMBOLS")
            for k, v in vars(m).items():
                print(k, "=", pformat(v))
        if plot:
            m.plot(show=True)

    for cmd, help in list(cls.OPTIONS.items())[::-1]:
        cmd, _, kind = cmd.partition(":")
        kind = kind_map[kind or "float"]
        cmd = cmd.replace("_", "-")
        cli = click.option(f"--{cmd}", help=help, type=kind)(cli)
    cli = click.command()(cli)
    cli()


def _main(cls, *args, **kwargs):
    return cls(*args, **kwargs)
