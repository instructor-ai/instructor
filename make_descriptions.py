import os
import yaml
from typing import Optional
import asyncio
from openai import AsyncOpenAI
import typer
from rich.console import Console
from rich.progress import Progress
from pydantic import BaseModel
import instructor

console = Console()
client = instructor.from_openai(AsyncOpenAI())


async def generate_front_matter(client: AsyncOpenAI, content: str) -> tuple[str, str]:
    """
    Generate a title and description for the given content using AI.

    Args:
        client (AsyncOpenAI): The AsyncOpenAI client.
        content (str): The content of the file.

    Returns:
        tuple[str, str]: A tuple containing the generated title and description.
    """

    class TitleDescription(BaseModel):
        title: str
        description: str

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an AI assistant that generates concise and relevant titles and descriptions for markdown files.",
            },
            {"role": "user", "content": content},
            {
                "role": "user",
                "content": "Based on the content, generate a concise title and a brief description (max 160 characters) that would be suitable for SEO purposes.",
            },
        ],
        max_tokens=100,
        response_model=TitleDescription,
    )
    return response.title, response.description


async def process_file(client: AsyncOpenAI, file_path: str) -> None:
    """
    Process a single file, adding front matter if it doesn't exist.

    Args:
        client (AsyncOpenAI): The AsyncOpenAI client.
        file_path (str): The path to the file to process.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    if not content.startswith("---"):
        title, description = await generate_front_matter(client, content)
        front_matter = f"---\ntitle: {title}\ndescription: {description}\n---\n\n"
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(front_matter + content)
        console.print(f"[green]Added front matter to {file_path}[/green]")
    else:
        console.print(f"[yellow]Front matter already exists in {file_path}[/yellow]")


async def process_files(root_dir: str, api_key: Optional[str] = None) -> None:
    """
    Process all markdown files in the given directory and its subdirectories.

    Args:
        root_dir (str): The root directory to start processing from.
        api_key (Optional[str]): The OpenAI API key. If not provided, it will be read from the OPENAI_API_KEY environment variable.
    """
    markdown_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".md"):
                markdown_files.append(os.path.join(root, file))

    with Progress() as progress:
        task = progress.add_task(
            "[green]Processing files...", total=len(markdown_files)
        )

        for file_path in markdown_files:
            await process_file(client, file_path)
            progress.update(task, advance=1)

    console.print("[bold green]All files processed successfully![/bold green]")


app = typer.Typer()


@app.command()
def main(
    root_dir: str = typer.Option("docs", help="Root directory to process"),
    api_key: Optional[str] = typer.Option(None, help="OpenAI API key"),
):
    """
    Add front matter to markdown files in the given directory and its subdirectories.
    """
    asyncio.run(process_files(root_dir, api_key))


if __name__ == "__main__":
    app()
