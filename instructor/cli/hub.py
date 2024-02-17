from typing import Iterable, Annotated, Optional
import typer
import httpx
import yaml

from openai import BaseModel, OpenAI
from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown

app = typer.Typer(
    name="hub",
    help="Interact with the instructor hub, a collection of examples and cookbooks for the instructor library.",
    short_help="Interact with the instructor hub",
)
client = OpenAI()
console = Console()


class Cookbook(BaseModel):
    id: int
    branch: str = "main"
    slug: str
    title: str

    def get_doc_url(self):
        """
        Returns the URL for the documentation
        """
        return f"https://jxnl.github.io/instructor/hub/{self.slug}/"

    def get_md_url(self):
        """
        Returns the raw URL for the markdown file
        """
        return f"https://raw.githubusercontent.com/jxnl/instructor/{self.branch}/docs/hub/{self.slug}.md?raw=true"

    def render_doc_link(self):
        """
        Rich Render the documentation link
        """
        return f"[link={self.get_doc_url()}](doc)[/link]"

    def render_slug(self):
        """
        Rich Render the slug with a clickable link to the documentation
        """
        return f"{self.slug} {self.render_doc_link()}"

    def get_md(self):
        url = self.get_md_url()
        resp = httpx.get(url)
        return resp.content.decode("utf-8")

    def get_py(self):
        """
        Uses a regex to find all ```python or ```py code blocks in the markdown file
        concatenates them and returns the code as a string
        """
        import re

        url = self.get_md_url()
        resp = httpx.get(url)
        script_str = resp.content.decode("utf-8")

        code_blocks = re.findall(r"```(python|py)(.*?)```", script_str, re.DOTALL)
        code = "\n".join([code_block for (_, code_block) in code_blocks])
        return code


def get_root_url(branch="main") -> str:
    """
    Returns the raw URL for the mkdocs.yml file
    """
    return f"https://raw.githubusercontent.com/jxnl/instructor/{branch}/mkdocs.yml?raw=true"


def list_hub(branch="main") -> Iterable[Cookbook]:
    url = get_root_url(branch)
    resp = httpx.get(url)
    # Fix for yaml parser
    mkdocs_config = resp.content.decode("utf-8").replace("!", "")
    data = yaml.safe_load(mkdocs_config)

    # Replace with Hub key
    cookbooks = [obj["Hub"] for obj in data.get("nav", []) if "Hub" in obj][0]
    for ii, cookbook in enumerate(cookbooks):
        title, link = list(cookbook.items())[0]
        slug = link.split("/")[-1].replace(".md", "")
        if slug != "index":
            yield Cookbook(id=ii, branch=branch, slug=slug, title=title)


def get_cookbook_by_id(id: int, branch="main"):
    for cookbook in list_hub(branch):
        if cookbook.id == id:
            return cookbook
    return None


def get_cookbook_by_slug(slug: str, branch="main"):
    for cookbook in list_hub(branch):
        if cookbook.slug == slug:
            return cookbook
    return None


@app.command(
    "list",
    help="List all available cookbooks",
    short_help="List all available cookbooks",
)
def list_cookbooks(
    branch: Annotated[
        str,
        typer.Option(
            "--branch",
            "-b",
            help="Specific branch to fetch the cookbooks from. Defaults to 'main'.",
        ),
    ] = "hub",
):
    table = Table(title="Available Cookbooks")
    table.add_column("hub_id", justify="right", style="cyan", no_wrap=True)
    table.add_column("slug", style="green")
    table.add_column("title", style="white")

    for cookbook in list_hub(branch):
        ii = cookbook.id
        slug = cookbook.render_slug()
        title = cookbook.title
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
    """Pull the latest cookbooks from the instructor hub based on id or slug.

    Args:
        id: The cookbook id. Optional.
        slug: The cookbook slug. Optional.
        py: Output to a Python file. Defaults to False.
        branch: Specific branch to fetch the cookbooks from. Defaults to 'main'.
        page: Paginate the output with a less-like pager. Defaults to False.
    """
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

    output = cookbook.get_py() if py else Markdown(cookbook.get_md())
    if page:
        with console.pager(styles=True):
            console.print(output)
    else:
        console.print(output)
