import os
from rich.console import Console
from rich.table import Table
from rich.live import Live
import typer
import time
import json
from typing import Any, Optional
from instructor.batch import BatchProcessor
from instructor.auto_client import from_provider
import warnings

from tqdm import tqdm

app = typer.Typer()

console = Console()


def generate_table(batch_jobs: list[Any], provider: str):
    """Generate table for batch jobs based on provider"""
    table = Table(title=f"{provider.title()} Batch Jobs")

    table.add_column("Batch ID", style="dim", min_width=30, no_wrap=True)
    table.add_column("Created At")
    table.add_column("Status")

    # Add provider-specific columns
    if provider == "openai":
        table.add_column("Failed")
        table.add_column("Completed")
        table.add_column("Total")
    elif provider == "anthropic":
        table.add_column("Request Count")
    elif provider == "google":
        table.add_column("State")

    for batch_job in batch_jobs:
        if provider == "openai":
            table.add_row(
                str(batch_job.id),
                str(batch_job.created_at),
                str(batch_job.status),
                str(getattr(batch_job.request_counts, "failed", "N/A")),
                str(getattr(batch_job.request_counts, "completed", "N/A")),
                str(getattr(batch_job.request_counts, "total", "N/A")),
            )
        elif provider == "anthropic":
            table.add_row(
                str(batch_job.id),
                str(batch_job.created_at),
                str(batch_job.processing_status),
                str(
                    getattr(batch_job.request_counts, "processing", "N/A")
                    if hasattr(batch_job, "request_counts")
                    else "N/A"
                ),
            )
        elif provider == "google":
            table.add_row(
                str(getattr(batch_job, "name", batch_job.id)),
                str(getattr(batch_job, "create_time", "N/A")),
                str(getattr(batch_job, "state", "N/A")),
                str(getattr(batch_job, "state", "N/A")),
            )

    return table


def get_jobs(limit: int = 10, provider: str = "openai"):
    """Get batch jobs for the specified provider"""

    if provider == "openai":
        from openai import OpenAI

        client = OpenAI()
        return client.batches.list(limit=limit).data

    elif provider == "anthropic":
        from anthropic import Anthropic

        client = Anthropic()
        # TODO: Remove beta fallback when stable API is available
        try:
            batches_client = client.messages.batches
        except AttributeError:
            batches_client = client.beta.messages.batches
        response = batches_client.list(limit=limit)
        return response.data

    elif provider == "google":
        from google import genai
        from google.genai.types import HttpOptions

        client = genai.Client(http_options=HttpOptions(api_version="v1"))
        try:
            # List batch jobs for Google GenAI
            jobs = client.batches.list(limit=limit)
            return list(jobs)
        except Exception as e:
            console.print(f"[red]Error listing Google batch jobs: {e}[/red]")
            return []

    else:
        raise ValueError(f"Unsupported provider: {provider}")


@app.command(name="list", help="See all existing batch jobs")
def watch(
    limit: int = typer.Option(10, help="Total number of batch jobs to show"),
    poll: int = typer.Option(
        10, help="Time in seconds to wait for the batch job to complete"
    ),
    screen: bool = typer.Option(False, help="Enable or disable screen output"),
    live: bool = typer.Option(
        False, help="Enable live polling to continuously update the table"
    ),
    provider: str = typer.Option(
        "openai",
        help="Provider to use (e.g., 'openai', 'anthropic', 'google')",
    ),
    # Deprecated flag for backward compatibility
    use_anthropic: bool = typer.Option(
        None,
        help="[DEPRECATED] Use --model instead. Use Anthropic API instead of OpenAI",
    ),
):
    """
    Monitor the status of the most recent batch jobs
    """
    # Handle deprecated flag
    if use_anthropic is not None:
        warnings.warn(
            "--use-anthropic is deprecated. Use --provider 'anthropic' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if use_anthropic:
            provider = "anthropic"

    # Check if required API key is available for the provider
    required_keys = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
    }

    if provider in required_keys and not os.getenv(required_keys[provider]):
        console.print(
            f"[red]Error: {required_keys[provider]} environment variable not set for {provider}[/red]"
        )
        return

    batch_jobs = get_jobs(limit, provider)
    table = generate_table(batch_jobs, provider)

    if not live:
        # Show table once and exit
        console.print(table)
        return

    # Live polling mode
    with Live(table, refresh_per_second=2, screen=screen) as live_table:
        while True:
            batch_jobs = get_jobs(limit, provider)
            table = generate_table(batch_jobs, provider)
            live_table.update(table)
            time.sleep(poll)


@app.command(
    help="Create a batch job from a file",
)
def create_from_file(
    file_path: str = typer.Option(help="File containing the batch job requests"),
    model: str = typer.Option(
        "openai/gpt-4o-mini",
        help="Model in format 'provider/model-name' (e.g., 'openai/gpt-4', 'anthropic/claude-3-sonnet')",
    ),
    # Deprecated flag for backward compatibility
    use_anthropic: bool = typer.Option(
        None,
        help="[DEPRECATED] Use --model instead. Use Anthropic API instead of OpenAI",
    ),
):
    # Handle deprecated flag
    if use_anthropic is not None:
        warnings.warn(
            "--use-anthropic is deprecated. Use --model 'anthropic/claude-3-sonnet' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if use_anthropic:
            model = "anthropic/claude-3-sonnet"

    provider, _ = model.split("/", 1)

    if provider == "anthropic":
        from anthropic import Anthropic

        client = Anthropic()
        with console.status(
            "[bold green]Creating Anthropic batch job...", spinner="dots"
        ):
            with open(file_path) as file:
                requests = [json.loads(line) for line in file]

            # TODO: Remove beta fallback when stable API is available
            try:
                batches_client = client.messages.batches
            except AttributeError:
                batches_client = client.beta.messages.batches
            batch = batches_client.create(requests=requests)
        console.print(f"Anthropic batch job created with ID: {batch.id}")
    else:
        from openai import OpenAI

        client = OpenAI()
        with console.status(
            f"[bold green] Uploading batch job file...", spinner="dots"
        ):
            batch_input_file = client.files.create(
                file=open(file_path, "rb"), purpose="batch"
            )

        batch_input_file_id = batch_input_file.id

        with console.status(
            f"[bold green] Creating batch job from ID {batch_input_file_id}",
            spinner="dots",
        ):
            client.batches.create(
                input_file_id=batch_input_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={"description": "testing job"},
            )

    # Skip the watch command to avoid timeout issues in testing


@app.command(help="Cancel a batch job")
def cancel(
    batch_id: str = typer.Option(help="Batch job ID to cancel"),
    use_anthropic: bool = typer.Option(
        False, help="Use Anthropic API instead of OpenAI"
    ),
):
    try:
        if use_anthropic:
            from anthropic import Anthropic

            client = Anthropic()
            # TODO: Remove beta fallback when stable API is available
            try:
                batches_client = client.messages.batches
            except AttributeError:
                batches_client = client.beta.messages.batches
            batches_client.cancel(batch_id)
        else:
            from openai import OpenAI

            client = OpenAI()
            client.batches.cancel(batch_id)
        watch(limit=5, poll=2, screen=False, live=False, use_anthropic=use_anthropic)
        console.log(f"[bold red]Job {batch_id} cancelled successfully!")
    except Exception as e:
        console.log(f"[bold red]Error cancelling job {batch_id}: {e}")


@app.command(help="Download the file associated with a batch job")
def download_file(
    batch_id: str = typer.Option(help="Batch job ID to download"),
    download_file_path: str = typer.Option(help="Path to download file to"),
    provider: str = typer.Option(
        "openai",
        help="Provider to use (e.g., 'openai', 'anthropic', 'google')",
    ),
):
    try:
        if provider == "anthropic":
            from anthropic import Anthropic

            client = Anthropic()
            # TODO: Remove beta fallback when stable API is available
            try:
                batches_client = client.messages.batches
            except AttributeError:
                batches_client = client.beta.messages.batches
            batch = batches_client.retrieve(batch_id)
            if batch.processing_status != "ended":
                raise ValueError("Only completed Jobs can be downloaded")

            results_url = batch.results_url
            if not results_url:
                raise ValueError("Results URL not available")

            with open(download_file_path, "w") as file:
                for result in tqdm(client.messages.batches.results(batch_id)):
                    file.write(json.dumps(result.model_dump()) + "\n")
        else:
            from openai import OpenAI

            client = OpenAI()
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


@app.command(help="Retrieve results from a batch job")
def results(
    batch_id: str = typer.Option(help="Batch job ID to get results from"),
    output_file: str = typer.Option(help="File to save the results to"),
    model: str = typer.Option(
        "openai/gpt-4o-mini",
        help="Model in format 'provider/model-name' (e.g., 'openai/gpt-4', 'anthropic/claude-3-sonnet')",
    ),
):
    """Retrieve and save batch job results"""
    provider, _ = model.split("/", 1)

    try:
        if provider == "openai":
            from openai import OpenAI

            client = OpenAI()
            batch = client.batches.retrieve(batch_id=batch_id)

            if batch.status != "completed":
                console.print(
                    f"[yellow]Batch status is '{batch.status}', not completed[/yellow]"
                )
                return

            file_id = batch.output_file_id
            if not file_id:
                console.print("[red]No output file available[/red]")
                return

            file_response = client.files.content(file_id)
            with open(output_file, "w") as f:
                f.write(file_response.text)
            console.print(f"[bold green]Results saved to: {output_file}[/bold green]")

        elif provider == "anthropic":
            from anthropic import Anthropic

            client = Anthropic()
            batch = client.beta.messages.batches.retrieve(batch_id)

            if batch.processing_status != "ended":
                console.print(
                    f"[yellow]Batch status is '{batch.processing_status}', not ended[/yellow]"
                )
                return

            # Get results from Anthropic batch API
            results_iter = client.beta.messages.batches.results(batch_id)

            with open(output_file, "w") as f:
                for result in results_iter:
                    f.write(json.dumps(result.model_dump()) + "\n")
            console.print(f"[bold green]Results saved to: {output_file}[/bold green]")

        elif provider == "google":
            console.print(
                "[red]Google/Gemini batch results via CLI not yet implemented[/red]"
            )
            console.print(
                "[yellow]Check your Google Cloud Storage bucket for results[/yellow]"
            )

        else:
            console.print(f"[red]Unsupported provider: {provider}[/red]")

    except Exception as e:
        console.log(f"[bold red]Error retrieving results for {batch_id}: {e}")


@app.command(help="Create batch job using BatchProcessor")
def create(
    messages_file: str = typer.Option(help="JSONL file with message conversations"),
    model: str = typer.Option(
        "openai/gpt-4o-mini",
        help="Model in format 'provider/model-name' (e.g., 'openai/gpt-4', 'anthropic/claude-3-sonnet')",
    ),
    response_model: str = typer.Option(
        help="Python class path for response model (e.g., 'examples.User')"
    ),
    output_file: str = typer.Option(
        "batch_requests.jsonl", help="Output file for batch requests"
    ),
    max_tokens: int = typer.Option(1000, help="Maximum tokens per request"),
    temperature: float = typer.Option(0.1, help="Temperature for generation"),
):
    """Create a batch job using the unified BatchProcessor"""
    try:
        # Import the response model dynamically
        module_path, class_name = response_model.rsplit(".", 1)
        import importlib

        module = importlib.import_module(module_path)
        response_class = getattr(module, class_name)

        # Load messages from file
        messages_list = []
        with open(messages_file, "r") as f:
            for line in f:
                if line.strip():
                    messages_list.append(json.loads(line))

        # Create batch processor
        processor = BatchProcessor(model, response_class)

        # Create batch file
        with console.status(
            f"[bold green]Creating batch file with {len(messages_list)} requests...",
            spinner="dots",
        ):
            processor.create_batch_from_messages(
                messages_list, output_file, max_tokens, temperature
            )

        console.print(f"[bold green]Batch file created: {output_file}[/bold green]")
        console.print(
            f"[yellow]Use 'instructor batch create-from-file --file-path {output_file}' to submit the batch[/yellow]"
        )

    except Exception as e:
        console.log(f"[bold red]Error creating batch: {e}")
