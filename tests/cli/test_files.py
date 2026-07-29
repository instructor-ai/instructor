"""Tests for instructor.cli.files."""

import os

os.environ.setdefault("OPENAI_API_KEY", "test-key-for-cli-import")

from openai.types import FileObject

from instructor.cli.files import generate_file_table


def test_generate_file_table_uses_attribute_access():
    """FileObject (and other OpenAI SDK response objects) are Pydantic
    models, not dicts — generate_file_table must read fields via attribute
    access, not `file["id"]`-style subscripting, which raises TypeError."""
    file = FileObject(
        id="file-abc123",
        bytes=1024,
        created_at=1700000000,
        filename="training.jsonl",
        object="file",
        purpose="fine-tune",
        status="processed",
    )

    table = generate_file_table([file])

    rendered_first_column = table.columns[0]._cells
    assert list(rendered_first_column) == ["file-abc123"]
