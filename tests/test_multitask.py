from instructor import OpenAISchema
from instructor.dsl import Iterable


def test_multi_task():
    class Search(OpenAISchema):
        """This is the search docstring"""

        id: int
        query: str

    multitask = Iterable(Search)
    assert multitask.openai_schema["name"] == "MultiSearch"
    assert (
        multitask.openai_schema["description"]
        == "Correct segmentation of `Search` tasks"
    )
