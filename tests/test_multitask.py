from openai_function_call import OpenAISchema
from openai_function_call.dsl import MultiTask


def test_multi_task():
    class Search(OpenAISchema):
        """This is the search docstring"""

        id: int
        query: str

    multitask = MultiTask(Search)
    assert multitask.openai_schema == {
        "description": "Correct segmentation of `Search` tasks",
        "name": "MultiSearch",
        "parameters": {
            "$defs": {
                "Search": {
                    "properties": {
                        "id": {"type": "integer"},
                        "query": {"type": "string"},
                    },
                    "required": ["id", "query"],
                    "description": "This is the search docstring",
                    "type": "object",
                }
            },
            "properties": {
                "tasks": {
                    "description": "Correctly segmented list of `Search` tasks",
                    "items": {"$ref": "#/$defs/Search"},
                    "type": "array",
                }
            },
            "required": ["tasks"],
            "type": "object",
        },
    }


def test_multi_task_with_name_and_desc():
    class Search(OpenAISchema):
        """This is the search docstring"""

        id: int
        query: str

    multitask = MultiTask(
        subtask_class=Search, name="MyCustomName", description="MyCustomDesc"
    )
    assert multitask.openai_schema == {
        "description": "MyCustomDesc",
        "name": "MultiMyCustomName",
        "parameters": {
            "$defs": {
                "Search": {
                    "properties": {
                        "id": {"type": "integer"},
                        "query": {"type": "string"},
                    },
                    "required": ["id", "query"],
                    "description": "This is the search docstring",
                    "type": "object",
                }
            },
            "properties": {
                "tasks": {
                    "description": "Correctly segmented list of `MyCustomName` tasks",
                    "items": {"$ref": "#/$defs/Search"},
                    "type": "array",
                }
            },
            "required": ["tasks"],
            "type": "object",
        },
    }
