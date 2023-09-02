from typing import List
from datetime import datetime, timedelta
import typer
import os
import aiohttp
import asyncio
from collections import defaultdict
from rich.console import Console
from rich.table import Table
from rich.progress import Progress

app = typer.Typer()
console = Console()

api_key = os.environ.get("OPENAI_API_KEY")


async def fetch_usage(date: str) -> dict:
    headers = {"Authorization": f"Bearer {api_key}"}
    url = f"https://api.openai.com/v1/usage?date={date}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as resp:
            return await resp.json()


def generate_usage_table(usage_data: List[dict]) -> Table:
    table = Table(title="OpenAI API Usage")
    table.add_column("Timestamp", style="dim")
    table.add_column("Requests", justify="right")
    table.add_column("Snapshot ID")
    table.add_column("Context Tokens")
    table.add_column("Generated Tokens")

    for usage in usage_data:
        table.add_row(
            str(datetime.fromtimestamp(usage["aggregation_timestamp"])),
            str(usage["n_requests"]),
            usage["snapshot_id"],
            str(usage["n_context_tokens_total"]),
            str(usage["n_generated_tokens_total"]),
        )
    return table


async def get_usage_for_past_n_days(n_days: int) -> List[dict]:
    tasks = []
    all_data = []
    with Progress() as progress:
        task = progress.add_task("[green]Fetching usage data...", total=n_days)
        for i in range(n_days):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            tasks.append(fetch_usage(date))
            progress.update(task, advance=1)

        fetched_data = await asyncio.gather(*tasks)
        for data in fetched_data:
            all_data.extend(data.get("data", []))
    return all_data


def group_and_sum_by_date_and_snapshot(usage_data: List[dict]) -> Table:
    summary = defaultdict(
        lambda: defaultdict(lambda: {"total_requests": 0, "total_tokens": 0})
    )

    for usage in usage_data:
        snapshot_id = usage["snapshot_id"]
        date = datetime.fromtimestamp(usage["aggregation_timestamp"]).strftime(
            "%Y-%m-%d"
        )
        summary[date][snapshot_id]["total_requests"] += usage["n_requests"]
        summary[date][snapshot_id]["total_tokens"] += usage["n_generated_tokens_total"]

    table = Table(title="Usage Summary by Date and Snapshot")
    table.add_column("Date", style="dim")
    table.add_column("Snapshot ID", style="dim")
    table.add_column("Total Requests", justify="right")
    table.add_column("Total Tokens", justify="right")

    for date, snapshots in summary.items():
        for snapshot_id, data in snapshots.items():
            table.add_row(
                date,
                snapshot_id,
                str(data["total_requests"]),
                str(data["total_tokens"]),
            )

    return table


@app.command(help="Displays OpenAI API usage data for the past N days.")
def list(
    n: int = typer.Option(5, help="Number of days."),
):
    all_data = asyncio.run(get_usage_for_past_n_days(n))
    table = group_and_sum_by_date_and_snapshot(all_data)
    console.print(table)


@app.command(
    help="Groups the OpenAI API usage data by snapshot_id and sums the total tokens."
)
def usage_group_by_snapshot():
    usage_data = asyncio.run(fetch_usage(datetime.now().strftime("%Y-%m-%d")))
    table = group_and_sum_by_snapshot(usage_data.get("data", []))
    console.print(table)


if __name__ == "__main__":
    app()
