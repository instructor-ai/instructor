import instructor
from typing import List
from openai import AsyncOpenAI
from asyncio import run
from tqdm.asyncio import tqdm_asyncio as asyncio
from pydantic import BaseModel, Field
import sqlite3


client = instructor.from_openai(AsyncOpenAI())


class TodoItem(BaseModel):
    """
    This is a schema that represents an actionable item which the user needs to consider
    """

    title: str = Field(..., description="This is a title for the todo item")
    description: str = Field(
        ...,
        description="This is a description that explains a plan of action for the todo",
    )


async def extract_todo(user_query: str):
    res = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a world class system that excels at extracting todo items from a user query",
            },
            {"role": "user", "content": user_query},
        ],
        response_model=List[TodoItem],
    )
    return [(item, user_query) for item in res]


async def process_todos(items):
    coros = [extract_todo(item) for item in items]
    results = await asyncio.gather(*coros)
    return [item for sublist in results for item in sublist]


if __name__ == "__main__":
    con = sqlite3.connect("tutorial.db")
    cur = con.cursor()
    cur.execute(
        "CREATE TABLE IF NOT EXISTS todos(id INTEGER PRIMARY KEY AUTOINCREMENT, annotated BOOLEAN DEFAULT FALSE, title TEXT, description TEXT, original_prompt TEXT)"
    )

    data = [
        "This week I need to finalize the project report, schedule a meeting with the team, prepare the presentation slides, submit the budget review, and send the client update emails.",
        "Next week I must organize the department outing, update the project timeline, review the new intern applications, and coordinate the quarterly webinars.",
        "Tomorrow I should finalize the contract details, call the supplier for an update, draft the monthly newsletter, and check the inventory status.",
        "By the end of this month, I need to complete the performance reviews, plan the training sessions, archive old project files, and renew the software licenses.",
        "This Friday I have to prepare the weekly sales report, confirm the client appointments, oversee the network upgrade, and document the audit findings.",
    ]

    todos: List[TodoItem] = run(process_todos(data))

    with sqlite3.connect("tutorial.db") as con:
        cur = con.cursor()
        for todo, original_query in todos:
            cur.execute(
                "INSERT INTO todos (title, description,original_prompt) VALUES (?, ?,?)",
                (todo.title, todo.description, original_query),
            )
        con.commit()
