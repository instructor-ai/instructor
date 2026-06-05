from __future__ import annotations

from typing import Any, cast

from pydantic import BaseModel

from instructor.v2.dsl.iterable import IterableBase, IterableModel


class User(BaseModel):
    name: str
    bio: str


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
