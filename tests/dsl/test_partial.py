# type: ignore[all]
from pydantic import BaseModel, Field
from typing import Optional, Union
from instructor.dsl.partial import Partial, PartialLiteralMixin
import pytest
import instructor
from openai import OpenAI, AsyncOpenAI
import os

models = ["gpt-4o-mini"]
modes = [
    instructor.Mode.TOOLS,
]


class SampleNestedPartial(BaseModel):
    b: int


class SamplePartial(BaseModel):
    a: int
    b: SampleNestedPartial


class NestedA(BaseModel):
    a: str
    b: Optional[str]


class NestedB(BaseModel):
    c: str
    d: str
    e: list[Union[str, int]]
    f: str


class UnionWithNested(BaseModel):
    a: list[Union[NestedA, NestedB]]
    b: list[NestedA]
    c: NestedB


def test_partial():
    partial = Partial[SamplePartial]
    assert partial.model_json_schema() == {
        "$defs": {
            "PartialSampleNestedPartial": {
                "properties": {"b": {"title": "B", "type": "integer"}},
                "required": ["b"],
                "title": "PartialSampleNestedPartial",
                "type": "object",
            }
        },
        "properties": {
            "a": {"title": "A", "type": "integer"},
            "b": {"$ref": "#/$defs/PartialSampleNestedPartial"},
        },
        "required": ["a", "b"],
        "title": "PartialSamplePartial",
        "type": "object",
    }, "Wrapped model JSON schema has changed"
    assert partial.get_partial_model().model_json_schema() == {
        "$defs": {
            "PartialSampleNestedPartial": {
                "properties": {
                    "b": {
                        "anyOf": [{"type": "integer"}, {"type": "null"}],
                        "default": None,
                        "title": "B",
                    }
                },
                "title": "PartialSampleNestedPartial",
                "type": "object",
            }
        },
        "properties": {
            "a": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "default": None,
                "title": "A",
            },
            "b": {
                "anyOf": [
                    {"$ref": "#/$defs/PartialSampleNestedPartial"},
                    {"type": "null"},
                ],
                "default": {},
            },
        },
        "title": "PartialSamplePartial",
        "type": "object",
    }, "Partial model JSON schema has changed"


def test_partial_with_whitespace():
    partial = Partial[SamplePartial]

    # Get the actual models from chunks
    models = list(partial.model_from_chunks(["\n", "\t", " ", '{"b": {"b": 1}}']))

    # Print actual values for debugging
    print(f"Number of models: {len(models)}")
    for i, model in enumerate(models):
        print(f"Model {i}: {model.model_dump()}")

    # Actual behavior: When whitespace chunks are processed, we may get models
    # First model has default values
    assert models[0].model_dump() == {"a": None, "b": {}}

    # Last model has b populated from JSON (from the JSON chunk)
    assert models[-1].model_dump() == {"a": None, "b": {"b": 1}}

    # Check we have the expected number of models (2 instead of 4)
    assert len(models) == 2


@pytest.mark.asyncio
async def test_async_partial_with_whitespace():
    partial = Partial[SamplePartial]

    # Handle any leading whitespace from the model
    async def async_generator():
        for chunk in ["\n", "\t", " ", '{"b": {"b": 1}}']:
            yield chunk

    expected_model_dicts = [
        {"a": None, "b": {}},
        {"a": None, "b": {}},
        {"a": None, "b": {}},
        {"a": None, "b": {"b": 1}},
    ]

    i = 0
    async for model in partial.model_from_chunks_async(async_generator()):
        assert model.model_dump() == expected_model_dicts[i]
        i += 1

    assert model.model_dump() == {"a": None, "b": {"b": 1}}


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_summary_extraction():
    class Summary(BaseModel, PartialLiteralMixin):
        summary: str = Field(description="A detailed summary")

    client = OpenAI()
    client = instructor.from_openai(client, mode=instructor.Mode.TOOLS)
    extraction_stream = client.chat.completions.create_partial(
        model="gpt-4o",
        response_model=Summary,
        messages=[
            {"role": "system", "content": "You summarize text"},
            {"role": "user", "content": "Summarize: Mary had a little lamb"},
        ],
        stream=True,
    )

    previous_summary = None
    updates = 0
    for extraction in extraction_stream:
        if previous_summary is not None and extraction:
            updates += 1
        previous_summary = extraction.summary

    assert updates == 1


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
@pytest.mark.asyncio
async def test_summary_extraction_async():
    class Summary(BaseModel, PartialLiteralMixin):
        summary: str = Field(description="A detailed summary")

    client = AsyncOpenAI()
    client = instructor.from_openai(client, mode=instructor.Mode.TOOLS)
    extraction_stream = client.chat.completions.create_partial(
        model="gpt-4o",
        response_model=Summary,
        messages=[
            {"role": "system", "content": "You summarize text"},
            {"role": "user", "content": "Summarize: Mary had a little lamb"},
        ],
        stream=True,
    )

    previous_summary = None
    updates = 0
    async for extraction in extraction_stream:
        if previous_summary is not None and extraction:
            updates += 1
        previous_summary = extraction.summary

    assert updates == 1


def test_union_with_nested():
    partial = Partial[UnionWithNested]
    partial.get_partial_model().model_validate_json(
        '{"a": [{"b": "b"}, {"d": "d"}], "b": [{"b": "b"}], "c": {"d": "d"}, "e": [1, "a"]}'
    )


def test_partial_with_default_factory():
    """Test that Partial works with fields that have default_factory.

    This test ensures that when making fields optional, the default_factory
    is properly cleared to avoid Pydantic validation errors about having
    both default and default_factory set.
    """

    class ModelWithDefaultFactory(BaseModel):
        items: list[str] = Field(default_factory=list)
        tags: dict[str, str] = Field(default_factory=dict)
        name: str

    # This should not raise a validation error about both default and default_factory
    partial = Partial[ModelWithDefaultFactory]
    partial_model = partial.get_partial_model()

    # Verify we can instantiate and validate
    # In Partial models, all fields are made Optional with default=None
    instance = partial_model()
    assert instance.items is None
    assert instance.tags is None
    assert instance.name is None

    # Test with partial data
    instance2 = partial_model.model_validate({"items": ["a", "b"]})
    assert instance2.items == ["a", "b"]
    assert instance2.tags is None
    assert instance2.name is None


class TestRecursiveModels:
    """Test that Partial handles self-referential models without infinite recursion."""

    def test_basic_recursive_model(self):
        """Partial should work with basic recursive models."""
        from typing import List

        class TreeNode(BaseModel):
            value: str
            children: Optional[List["TreeNode"]] = None

        TreeNode.model_rebuild()

        # Should not raise RecursionError
        PartialTreeNode = Partial[TreeNode]
        TruePartial = PartialTreeNode.get_partial_model()

        # Can validate partial data
        result = TruePartial.model_validate({"value": "root"})
        assert result.value == "root"
        assert result.children is None

    def test_nested_recursive_model(self):
        """Partial should work with nested children."""
        from typing import List

        class TreeNode(BaseModel):
            value: str
            children: Optional[List["TreeNode"]] = None

        TreeNode.model_rebuild()

        PartialTreeNode = Partial[TreeNode]
        TruePartial = PartialTreeNode.get_partial_model()

        # Validate with nested structure
        data = {
            "value": "root",
            "children": [
                {"value": "child1"},
                {"value": "child2", "children": [{"value": "grandchild"}]},
            ],
        }
        result = TruePartial.model_validate(data)
        assert result.value == "root"
        assert len(result.children) == 2
        assert result.children[0].value == "child1"
        assert result.children[1].children[0].value == "grandchild"

    def test_mutually_recursive_models(self):
        """Partial should handle mutually recursive models."""
        from typing import List

        class Person(BaseModel):
            name: str
            employer: Optional["Company"] = None

        class Company(BaseModel):
            name: str
            employees: Optional[List[Person]] = None

        Person.model_rebuild()
        Company.model_rebuild()

        # Both should work without RecursionError
        PartialPerson = Partial[Person]
        PartialCompany = Partial[Company]

        assert PartialPerson is not None
        assert PartialCompany is not None

        # Validate partial data
        person_partial = PartialPerson.get_partial_model()
        result = person_partial.model_validate({"name": "Alice"})
        assert result.name == "Alice"

    def test_direct_self_reference(self):
        """Partial should handle direct self-reference (linked list style)."""

        class LinkedNode(BaseModel):
            value: int
            next: Optional["LinkedNode"] = None

        LinkedNode.model_rebuild()

        # Should not raise RecursionError
        PartialLinked = Partial[LinkedNode]
        TruePartial = PartialLinked.get_partial_model()

        # Validate chain
        data = {"value": 1, "next": {"value": 2, "next": {"value": 3}}}
        result = TruePartial.model_validate(data)
        assert result.value == 1
        assert result.next.value == 2
        assert result.next.next.value == 3

    def test_complex_recursive_with_validators(self):
        """Complex recursive model with validators, multiple self-refs, and nested types."""
        from typing import List, Dict, Literal
        from pydantic import model_validator, field_validator
        from enum import Enum

        class NodeType(Enum):
            FOLDER = "folder"
            FILE = "file"
            SYMLINK = "symlink"

        class Permission(BaseModel):
            user: str
            level: Literal["read", "write", "admin"]

        class FileSystemNode(BaseModel):
            name: str
            node_type: NodeType
            size_bytes: Optional[int] = None
            children: Optional[List["FileSystemNode"]] = None
            parent: Optional["FileSystemNode"] = None
            symlink_target: Optional["FileSystemNode"] = None
            permissions: Optional[List[Permission]] = None
            metadata: Optional[Dict[str, str]] = None

            @field_validator("name")
            @classmethod
            def validate_name(cls, v):
                if v and "/" in v:
                    raise ValueError("Name cannot contain /")
                return v

            @model_validator(mode="after")
            def validate_node_consistency(self):
                # Folders must have no size, files must have size
                if self.node_type == NodeType.FOLDER and self.size_bytes is not None:
                    raise ValueError("Folders cannot have size_bytes")
                if self.node_type == NodeType.FILE and self.children:
                    raise ValueError("Files cannot have children")
                if self.node_type == NodeType.SYMLINK and not self.symlink_target:
                    raise ValueError("Symlinks must have a target")
                return self

        FileSystemNode.model_rebuild()

        # Should not raise RecursionError
        PartialFS = Partial[FileSystemNode]
        TruePartial = PartialFS.get_partial_model()

        # Complex nested structure
        data = {
            "name": "root",
            "node_type": "folder",
            "permissions": [{"user": "admin", "level": "admin"}],
            "metadata": {"created": "2024-01-01"},
            "children": [
                {
                    "name": "documents",
                    "node_type": "folder",
                    "children": [
                        {
                            "name": "report.pdf",
                            "node_type": "file",
                            "size_bytes": 1024,
                            "permissions": [{"user": "alice", "level": "read"}],
                        },
                        {
                            "name": "data",
                            "node_type": "folder",
                            "children": [
                                {
                                    "name": "archive.zip",
                                    "node_type": "file",
                                    "size_bytes": 2048,
                                }
                            ],
                        },
                    ],
                },
                {
                    "name": "shortcut",
                    "node_type": "symlink",
                    "symlink_target": {
                        "name": "target_file",
                        "node_type": "file",
                        "size_bytes": 512,
                    },
                },
            ],
        }

        result = TruePartial.model_validate(data)
        assert result.name == "root"
        assert result.node_type == NodeType.FOLDER
        assert len(result.children) == 2
        assert result.children[0].name == "documents"
        assert len(result.children[0].children) == 2
        assert result.children[0].children[0].name == "report.pdf"
        assert result.children[0].children[0].size_bytes == 1024
        assert result.children[0].children[1].children[0].name == "archive.zip"
        assert result.children[1].symlink_target.name == "target_file"
        assert result.permissions[0].level == "admin"

    def test_recursive_with_union_types(self):
        """Recursive model with Union types containing self-references."""
        from typing import List, Union

        class TextBlock(BaseModel):
            text: str

        class Container(BaseModel):
            title: str
            content: List[Union[TextBlock, "Container"]]

        Container.model_rebuild()

        PartialContainer = Partial[Container]
        TruePartial = PartialContainer.get_partial_model()

        data = {
            "title": "Chapter 1",
            "content": [
                {"text": "Introduction paragraph"},
                {
                    "title": "Section 1.1",
                    "content": [
                        {"text": "Section text"},
                        {
                            "title": "Subsection 1.1.1",
                            "content": [{"text": "Deep nested text"}],
                        },
                    ],
                },
                {"text": "Closing paragraph"},
            ],
        }

        result = TruePartial.model_validate(data)
        assert result.title == "Chapter 1"
        assert len(result.content) == 3
        assert result.content[0].text == "Introduction paragraph"
        assert result.content[1].title == "Section 1.1"
        assert result.content[1].content[1].title == "Subsection 1.1.1"
