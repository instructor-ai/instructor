from rich.console import Console
from rich.table import Table
from rich.live import Live
from openai import OpenAI
from openai.types.batch import Batch
import typer
import datetime
import time

client = OpenAI()
app = typer.Typer()

console = Console()


def generate_table(batch_jobs: list[Batch]):
    table = Table(
        title="OpenAI Batch Jobs",
    )

    table.add_column("Batch ID", style="dim")
    table.add_column("Created At")
    table.add_column("Status")
    table.add_column("Failed")
    table.add_column("Completed")
    table.add_column("Total")

    for batch_job in batch_jobs:
        table.add_row(
            batch_job.id,
            str(datetime.datetime.fromtimestamp(batch_job.created_at)),
            batch_job.status,
            str(batch_job.request_counts.failed),  # type: ignore
            str(batch_job.request_counts.completed),  # type: ignore
            str(batch_job.request_counts.total),  # type: ignore
        )

    return table


def get_jobs(limit: int = 10):
    return client.batches.list(limit=limit).data


@app.command(name="list", help="See all existing batch jobs")
def watch(
    limit: int = typer.Option(10, help="Total number of batch jobs to show"),
    poll: int = typer.Option(
        10, help="Time in seconds to wait for the batch job to complete"
    ),
    screen: bool = typer.Option(False, help="Enable or disable screen output"),
):
    """
    Monitor the status of the most recent batch jobs
    """
    batch_jobs = get_jobs(limit)
    table = generate_table(batch_jobs)
    with Live(
        generate_table(batch_jobs), refresh_per_second=2, screen=screen
    ) as live_table:
        while True:
            batch_jobs = get_jobs(limit)
            table = generate_table(batch_jobs)
            live_table.update(table)
            time.sleep(poll)


@app.command(
    help="Create a batch job from a file",
)
def create_from_file(
    file_path: str = typer.Option(help="File containing the batch job requests"),
):
    with console.status(f"[bold green] Uploading batch job file...", spinner="dots"):
        batch_input_file = client.files.create(
            file=open(file_path, "rb"), purpose="batch"
        )

    batch_input_file_id = batch_input_file.id

    with console.status(
        f"[bold green] Creating batch job from ID {batch_input_file_id}", spinner="dots"
    ):
        client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": "testing job"},
        )

    watch(limit=5, poll=2, screen=False)


@app.command(help="Cancel a batch job")
def cancel(batch_id: str = typer.Option(help="Batch job ID to cancel")):
    try:
        client.batches.cancel(batch_id)
        watch(limit=5, poll=2, screen=False)
        console.log(f"[bold red]Job {batch_id} cancelled successfully!")
    except Exception as e:
        console.log(f"[bold red]Error cancelling job {batch_id}: {e}")


@app.command(help="Download the file associated with a batch job")
def download_file(
    batch_id: str = typer.Option(help="Batch job ID to cancel"),
    download_file_path: str = typer.Option(help="Path to download file to"),
):
    try:
        batch = client.batches.retrieve(batch_id=batch_id)
        status = batch.status

        if status != "completed":
            raise ValueError("Only completed Jobs can be downloaded")

        file_id = batch.output_file_id

        assert file_id, f"Equivalent Output File not found for {batch_id}"
        file_response = client.files.content(file_id)

        with open(download_file_path, "w") as file:
            file.write(file_response.text)

    except Exception as e:
        console.log(f"[bold red]Error downloading file for {batch_id}: {e}")
