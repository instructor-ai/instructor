"""Regression tests for IterableBase.get_object string-aware brace matching.

These are pure-function tests (no API calls): they exercise the streaming
object splitter used by ``create_iterable()`` to make sure braces that appear
inside JSON string values do not prematurely terminate an object.
"""

from __future__ import annotations

import json

from pydantic import BaseModel

from instructor.dsl.iterable import IterableBase, IterableModel


def test_get_object_ignores_braces_inside_strings():
    # The "}" inside the string value must not end the object early.
    obj, rest = IterableBase.get_object('{"bio": "a } b"}, {"bio": "next"}', 0)
    assert obj is not None
    assert json.loads(obj) == {"bio": "a } b"}
    # The remainder still contains the next object.
    assert '{"bio": "next"}' in rest


def test_get_object_handles_escaped_quote_before_brace():
    obj, _ = IterableBase.get_object(r'{"bio": "sa\"y } hi"}', 0)
    assert obj is not None
    assert json.loads(obj) == {"bio": 'sa"y } hi'}


def test_get_object_still_handles_nested_objects():
    obj, _ = IterableBase.get_object('{"a": {"b": 1}}, rest', 0)
    assert obj is not None
    assert json.loads(obj) == {"a": {"b": 1}}


def test_tasks_from_chunks_preserves_braces_in_string_values():
    class User(BaseModel):
        name: str
        bio: str

    multi = IterableModel(User)
    chunks = [
        '{"tasks": [',
        '{"name": "Alice", "bio": "happy :}"}',
        ', {"name": "Bob", "bio": "plain"}',
        "]}",
    ]
    users = list(multi.tasks_from_chunks(chunks))
    assert [(u.name, u.bio) for u in users] == [("Alice", "happy :}"), ("Bob", "plain")]
