from __future__ import annotations

from typing import Any, cast

import pytest
from pydantic import BaseModel

from instructor.v2.dsl.iterable import IterableBase, IterableModel


class User(BaseModel):
    name: str
    bio: str


class Task(BaseModel):
    name: str
    priority: int


def test_iterable_get_object_ignores_braces_inside_strings() -> None:
    obj, rest = IterableBase.get_object('{"bio": "a } b"},{"bio": "next"}', 0)

    assert obj == '{"bio": "a } b"}'
    assert rest == '{"bio": "next"}'


def test_iterable_get_object_handles_escaped_quotes_before_brace() -> None:
    obj, rest = IterableBase.get_object(
        '{"bio": "quote \\" } still string"},{"bio": "next"}',
        0,
    )

    assert obj == '{"bio": "quote \\" } still string"}'
    assert rest == '{"bio": "next"}'


def test_iterable_get_object_handles_even_backslashes_before_quote() -> None:
    obj, rest = IterableBase.get_object(
        r'{"bio": "ends \\"},{"bio": "next"}',
        0,
    )

    assert obj == r'{"bio": "ends \\"}'
    assert rest == '{"bio": "next"}'


def test_iterable_tasks_from_chunks_handles_braces_inside_strings() -> None:
    chunks = [
        '{"tasks": [',
        '{"name": "Alice", "bio": "happy :}"}',
        ', {"name": "Bob", "bio": "plain"}',
        "]}",
    ]

    iterable_model = cast(Any, IterableModel(User))
    users = list(iterable_model.tasks_from_chunks(chunks))

    assert users == [
        User(name="Alice", bio="happy :}"),
        User(name="Bob", bio="plain"),
    ]


def test_iterable_tasks_from_chunks_ignores_bracket_inside_preceding_string() -> None:
    """A literal "[" inside a string field emitted before the tasks array must
    not be mistaken for the start of the array.

    `tasks_from_chunks` decides the array has started with a naive
    `"[" in chunk` check. If any field preceding "tasks" in the response JSON
    contains a literal "[" in a string value (e.g. a free-text "note" field),
    that in-string bracket falsely triggers array-start detection, the buffer
    gets sliced from the wrong offset, and the real tasks are silently lost.
    """
    chunks = [
        '{"note": "check [priority] items first", ',
        '"tasks": [',
        '{"name": "alpha", "priority": 1}',
        ', {"name": "beta", "priority": 2}',
        "]}",
    ]

    iterable_model = cast(Any, IterableModel(Task))
    tasks = list(iterable_model.tasks_from_chunks(chunks))

    assert tasks == [
        Task(name="alpha", priority=1),
        Task(name="beta", priority=2),
    ]


@pytest.mark.asyncio
async def test_iterable_tasks_from_chunks_async_ignores_bracket_inside_preceding_string() -> (
    None
):
    async def _achunks():
        for chunk in [
            '{"note": "check [priority] items first", ',
            '"tasks": [',
            '{"name": "alpha", "priority": 1}',
            ', {"name": "beta", "priority": 2}',
            "]}",
        ]:
            yield chunk

    iterable_model = cast(Any, IterableModel(Task))
    tasks = [task async for task in iterable_model.tasks_from_chunks_async(_achunks())]

    assert tasks == [
        Task(name="alpha", priority=1),
        Task(name="beta", priority=2),
    ]
