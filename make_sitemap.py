import os
import asyncio
import yaml
from typing import Optional
from collections.abc import Generator
from openai import AsyncOpenAI
import typer
from rich.console import Console
from rich.progress import Progress
import hashlib
from asyncio import as_completed
import tenacity

console = Console()


def traverse_docs(
    root_dir: str = "docs",
) -> Generator[tuple[str, str, str], None, None]:
    """
    Recursively traverse the docs folder and yield the path, content, and content hash of each file.

    Args:
        root_dir (str): The root directory to start traversing from. Defaults to 'docs'.

    Yields:
        Tuple[str, str, str]: A tuple containing the relative path from 'docs', the file content, and the content hash.
    """
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".md"):  # Assuming we're only interested in Markdown files
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, root_dir)

                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                content_hash = hashlib.md5(content.encode()).hexdigest()
                yield relative_path, content, content_hash


@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
    retry=tenacity.retry_if_exception_type(Exception),
    before_sleep=lambda retry_state: console.print(
        f"[yellow]Retrying summarization... (Attempt {retry_state.attempt_number})[/yellow]"
    ),
)
async def summarize_content(client: AsyncOpenAI, path: str, content: str) -> str:
    """
    Summarize the content of a file with retry logic.

    Args:
        client (AsyncOpenAI): The AsyncOpenAI client.
        path (str): The path of the file.
        content (str): The content of the file.

    Returns:
        str: A summary of the content.

    Raises:
        Exception: If all retry attempts fail.
    """
    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that summarizes text.",
                },
                {"role": "user", "content": content},
                {
                    "role": "user",
                    "content": "Please summarize the content in a few sentences so they can be used for SEO. Include core ideas, objectives, and important details and key points and key words",
                },
            ],
            max_tokens=4000,
        )
        return response.choices[0].message.content
    except Exception as e:
        console.print(f"[bold red]Error summarizing {path}: {str(e)}[/bold red]")
        raise  # Re-raise the exception to trigger a retry


async def generate_sitemap(
    root_dir: str,
    output_file: str,
    api_key: Optional[str] = None,
    max_concurrency: int = 5,
) -> None:
    """
    Generate a sitemap from the given root directory.

    Args:
        root_dir (str): The root directory to start traversing from.
        output_file (str): The output file to save the sitemap.
        api_key (Optional[str]): The OpenAI API key. If not provided, it will be read from the OPENAI_API_KEY environment variable.
        max_concurrency (int): The maximum number of concurrent tasks. Defaults to 5.
    """
    client = AsyncOpenAI(api_key=api_key)

    # Load existing sitemap if it exists
    existing_sitemap: dict[str, dict[str, str]] = {}
    if os.path.exists(output_file):
        with open(output_file, encoding="utf-8") as sitemap_file:
            existing_sitemap = yaml.safe_load(sitemap_file) or {}

    sitemap_data: dict[str, dict[str, str]] = {}

    async def process_file(
        path: str, content: str, content_hash: str
    ) -> tuple[str, dict[str, str]]:
        if (
            path in existing_sitemap
            and existing_sitemap[path].get("hash") == content_hash
        ):
            return path, existing_sitemap[path]
        try:
            summary = await summarize_content(client, path, content)
            return path, {"summary": summary, "hash": content_hash}
        except Exception as e:
            console.print(
                f"[bold red]Failed to summarize {path} after multiple attempts: {str(e)}[/bold red]"
            )
            return path, {"summary": "Failed to generate summary", "hash": content_hash}

    files_to_process: list[tuple[str, str, str]] = list(traverse_docs(root_dir))
    total_files = len(files_to_process)

    with Progress() as progress:
        task = progress.add_task("[green]Processing files...", total=total_files)

        semaphore = asyncio.Semaphore(max_concurrency)

        async def bounded_process_file(*args):
            async with semaphore:
                return await process_file(*args)

        tasks = [
            bounded_process_file(path, content, content_hash)
            for path, content, content_hash in files_to_process
        ]

        for completed_task in as_completed(tasks):
            path, result = await completed_task
            sitemap_data[path] = result
            progress.update(task, advance=1)

            # Save intermediate results
            with open(output_file, "w", encoding="utf-8") as sitemap_file:
                yaml.dump(sitemap_data, sitemap_file, default_flow_style=False)

    console.print(
        f"[bold green]Sitemap has been generated and saved to {output_file}[/bold green]"
    )


app = typer.Typer()


@app.command()
def main(
    root_dir: str = typer.Option("docs", help="Root directory to traverse"),
    output_file: str = typer.Option("sitemap.yaml", help="Output file for the sitemap"),
    api_key: Optional[str] = typer.Option(None, help="OpenAI API key"),
    max_concurrency: int = typer.Option(5, help="Maximum number of concurrent tasks"),
):
    """
    Generate a sitemap from the given root directory.
    """
    asyncio.run(generate_sitemap(root_dir, output_file, api_key, max_concurrency))


if __name__ == "__main__":
    app()
