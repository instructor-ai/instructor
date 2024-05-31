import pyperclip
import typer
from generate import generate_artifact
from extract import extract_artifacts
from rich.console import Console
from rich.spinner import Spinner

app = typer.Typer()


@app.command()
def process_clipboard(filename: str = "clip"):
    console = Console()

    if filename == "clip":
        transcript = pyperclip.paste()
    else:
        with open(filename, "r") as f:
            transcript = f.read()

    console.print(f"Characters in clipboard: {len(transcript)}")
    console.print(f"Start of text: {transcript[:100]}...")

    with console.status("Extracting artifacts"):
        artifacts = extract_artifacts(transcript)

    for artifact in artifacts.artifacts:
        console.print(
            f"Extracted Artifact",
            {
                "type": artifact.artifact_type,
                "title": artifact.title,
            },
        )

    for artifact in artifacts.artifacts:
        with console.status(
            f"Generating artifact {artifact.artifact_type}: {artifact.title}"
        ):
            doc = generate_artifact(transcript, artifact)
            print(doc.title)
            print(doc.content)
            doc.save()


if __name__ == "__main__":
    app()
