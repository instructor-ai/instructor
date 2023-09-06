import sys
from pydantic import BaseModel
import typer
import importlib
import os

app = typer.Typer(
    name="schema-catalog",
    help="Manage Schema Catalogs",
)


@app.command(
    help="Upload a Schema Catalog to OpenAI's servers",
)
def upload(
    model: str,
):
    # Add the current working directory to sys.path
    sys.path.insert(0, os.getcwd())

    module_name, class_name = model.split(":")

    # Dynamically import the Pydantic model class
    module = importlib.import_module(module_name)
    ModelClass: BaseModel = getattr(module, class_name)

    # Create an instance of the dynamically loaded Pydantic model
    import json

    INSTRUCTOR_API_KEY = os.environ.get("INSTRUCTOR_API_KEY", None)
    typer.echo(
        f"Creating Schema Catalog for {model} using Instructor API Key {INSTRUCTOR_API_KEY}"
    )
    typer.echo(json.dumps(ModelClass.model_json_schema(), indent=2))


if __name__ == "__main__":
    app()
