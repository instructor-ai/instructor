import typer

app = typer.Typer()

@app.command()
def hub() -> None:
    """
    This command has been deprecated. The instructor hub is no longer available.
    Please refer to our cookbook examples at https://python.useinstructor.com/examples/
    """
    typer.echo("The instructor hub has been deprecated. Please refer to our cookbook examples at https://python.useinstructor.com/examples/")
    raise typer.Abort()

if __name__ == "__main__":
    app()
