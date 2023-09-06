import typer
import instructor.cli.jobs as jobs
import instructor.cli.files as files
import instructor.cli.usage as usage

app = typer.Typer(
    name="instructor-ft",
    help="A CLI for fine-tuning OpenAI's models",
)

app.add_typer(jobs.app, name="jobs", help="Monitor and create fine tuning jobs")
app.add_typer(files.app, name="files", help="Manage files on OpenAI's servers")
app.add_typer(usage.app, name="usage", help="Check OpenAI API usage data")
