import typer
import jobs
import files

app = typer.Typer(
    name="instructor-ft",
    help="A CLI for fine-tuning OpenAI's models",
)

app.add_typer(jobs.app, name="jobs", help="Monitor and create fine tuning jobs")
app.add_typer(files.app, name="files", help="Manage files on OpenAI's servers")

if __name__ == "__main__":
    app()
