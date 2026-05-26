import typer

app = typer.Typer(name="simulon", help="AI cluster simulator")

from simulon.cli.trace import trace_app  # noqa: E402

app.add_typer(trace_app, name="trace")

from simulon.cli.install import app as install_app  # noqa: E402

app.add_typer(
    install_app, name="install", help="Install third-party components (apex, deepgemm, m4)."
)

from simulon.cli.simulate import simulate  # noqa: E402

app.command()(simulate)
