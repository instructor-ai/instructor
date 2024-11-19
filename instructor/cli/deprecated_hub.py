from typing import Any
import typer
from typer import Typer, Exit, echo

app: Typer = typer.Typer()

@app.command()
def hub() -> None:
    """
    This command has been deprecated. The instructor hub is no longer available.
    Please refer to our cookbook examples at https://python.useinstructor.com/examples/
    """
    echo("The instructor hub has been deprecated. Please refer to our cookbook examples at https://python.useinstructor.com/examples/")
    raise Exit(1)

if __name__ == "__main__":
    app()
