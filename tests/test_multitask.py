from instructor import OpenAISchema
from instructor.dsl import IterableModel


def test_multi_task():
    class Search(OpenAISchema):
        """This is the search docstring"""

        id: int
        query: str

    IterableSearch = IterableModel(Search)
    assert IterableSearch.openai_schema["name"] == "IterableSearch"
    assert (
        IterableSearch.openai_schema["description"]
        == "Correct segmentation of `Search` tasks"
    )
