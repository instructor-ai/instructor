from pydoc import cli
from typing import Iterable, Optional

import typer
import httpx
import yaml

from pydantic import BaseModel
from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown

app = typer.Typer(
    name="hub",
    help="Interact with the instructor hub, a collection of examples and cookbooks for the instructor library.",
    short_help="Interact with the instructor hub",
)
console = Console()

import requests


class HubClient:
    def __init__(
        self, base_url: str = "https://instructor-hub-proxy.jason-a3f.workers.dev"
    ):
        self.base_url = base_url

    def get_cookbooks(self, branch):
        """Get collection index of cookbooks."""
        url = f"{self.base_url}/api/{branch}/items"
        response = httpx.get(url)
        if response.status_code == 200:
            return [HubPage(**page) for page in response.json()]
        else:
            raise Exception(f"Failed to fetch cookbooks: {response.status_code}")

    def get_content_markdown(self, branch, slug):
        """Get markdown content."""
        url = f"{self.base_url}/api/{branch}/items/{slug}/md"
        response = httpx.get(url)
        if response.status_code == 200:
            return response.text
        else:
            raise Exception(f"Failed to fetch markdown content: {response.status_code}")

    def get_content_python(self, branch, slug):
        """Get Python code blocks from content."""
        url = f"{self.base_url}/api/{branch}/items/{slug}/py"
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        else:
            raise Exception(f"Failed to fetch Python content: {response.status_code}")


class HubPage(BaseModel):
    id: int
    branch: str = "main"
    slug: str
    name: str

    def get_doc_url(self) -> str:
        return f"https://jxnl.github.io/instructor/hub/{self.slug}/"

    def get_md_url(self) -> str:
        return f"https://raw.githubusercontent.com/jxnl/instructor/{self.branch}/docs/hub/{self.slug}.md?raw=true"

    def render_doc_link(self) -> str:
        return f"[link={self.get_doc_url()}](doc)[/link]"

    def render_slug(self) -> str:
        return f"{self.slug} {self.render_doc_link()}"


def mkdoc_yaml_url(branch="main") -> str:
    return f"https://raw.githubusercontent.com/jxnl/instructor/{branch}/mkdocs.yml?raw=true"


def get_cookbook_by_id(id: int, branch="main"):
    client = HubClient()
    for cookbook in client.get_cookbooks(branch):
        if cookbook.id == id:
            return cookbook
    return None


def get_cookbook_by_slug(slug: str, branch="main"):
    client = HubClient()
    for cookbook in client.get_cookbooks(branch):
        if cookbook.slug == slug:
            return cookbook
    return None


@app.command(
    "list",
    help="List all available cookbooks",
    short_help="List all available cookbooks",
)
def list_cookbooks(
    branch: str = typer.Option(
        "hub",
        "--branch",
        "-b",
        help="Specific branch to fetch the cookbooks from. Defaults to 'main'.",
    ),
):
    table = Table(title="Available Cookbooks")
    table.add_column("hub_id", justify="right", style="cyan", no_wrap=True)
    table.add_column("slug", style="green")
    table.add_column("title", style="white")

    client = HubClient()
    for cookbook in client.get_cookbooks(branch):
        ii = cookbook.id
        slug = cookbook.render_slug()
        title = cookbook.name
        table.add_row(str(ii), slug, title)

    console.print(table)


@app.command(
    "pull",
    help="Pull the latest cookbooks from the instructor hub, optionally outputting to a file",
    short_help="Pull the latest cookbooks",
)
def pull(
    id: Optional[int] = typer.Option(None, "--id", "-i", help="The cookbook id"),
    slug: Optional[str] = typer.Option(None, "--slug", "-s", help="The cookbook slug"),
    py: bool = typer.Option(False, "--py", help="Output to a Python file"),
    branch: str = typer.Option(
        "hub", help="Specific branch to fetch the cookbooks from."
    ),
    page: bool = typer.Option(
        False, "--page", "-p", help="Paginate the output with a less-like pager"
    ),
):
    client = HubClient()
    cookbook = (
        get_cookbook_by_id(id, branch)
        if id
        else get_cookbook_by_slug(slug, branch)
        if slug
        else None
    )
    if not cookbook:
        typer.echo("Please provide a valid cookbook id or slug.")
        raise typer.Exit(code=1)

    output = (
        client.get_content_python(branch, cookbook.slug)
        if py
        else client.get_content_markdown(branch, cookbook.slug)
    )

    if page:
        with console.pager(styles=True):
            console.print(output)
    else:
        console.print(output)
