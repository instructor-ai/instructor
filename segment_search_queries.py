"""
This script is used to segment a request into multiple search queries and perform them asynchronously.
The `Search` class represents a single search query and has the `execute` method to perform the search.
The `MultiSearch` class represents multiple searches and has an `execute` method that runs all the 
searches concurrently using asyncio.
The `segment` function uses OpenAI's GPT-3 model to convert a given string into multiple search queries,
which are then run by calling the `execute` method of the returned `MultiSearch` object.

Examples:
>>> queries = segment(
...     "Please send me the video from last week about the investment case study and also documents about your GPDR policy?"
... )
>>> queries.execute()
# Expected output:
# >>> Searching for `Video` with query `investment case study` using `SearchType.VIDEO`
# >>> Searching for `Documents` with query `GPDR policy` using `SearchType.EMAIL`
"""

from openai_function_call import OpenAISchema
from pydantic import Field
from typing import List
from tenacity import retry, stop_after_attempt
import openai
import enum


class SearchType(str, enum.Enum):
    """Enumeration representing the types of searches that can be performed."""

    VIDEO = "video"
    EMAIL = "email"


class Search(OpenAISchema):
    """
    Class representing a single search query.

    Args:
        title (str): The title of the request.
        query (str): The query string to search for.
        type (SearchType): The type of search to perform.
    """

    title: str = Field(..., description="Title of the request")
    query: str = Field(..., description="Query to search for relevant content")
    type: SearchType = Field(..., description="Type of search")

    async def execute(self):
        import asyncio

        await asyncio.sleep(1)
        print(
            f"Searching for `{self.title}` with query `{self.query}` using `{self.type}`"
        )


class MultiSearch(OpenAISchema):
    """
    Class representing multiple search queries.

    Args:
        searches (List[Search]): The list of searches to perform.
    """

    searches: List[Search] = Field(..., description="List of searches")

    def execute(self):
        import asyncio

        loop = asyncio.get_event_loop()

        tasks = asyncio.gather(*[search.execute() for search in self.searches])
        return loop.run_until_complete(tasks)


@retry(stop=stop_after_attempt(3))
def segment(data: str) -> MultiSearch:
    """
    Convert a string into multiple search queries using OpenAI's GPT-3 model.

    Args:
        data (str): The string to convert into search queries.

    Returns:
        MultiSearch: An object representing the multiple search queries.
    """

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        temperature=0.1,
        functions=[MultiSearch.openai_schema],
        function_call={"name": MultiSearch.openai_schema["name"]},
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": f"Consider the data below:\n{data} and segment it into multiple search queries",
            },
        ],
        max_tokens=1000,
    )
    return MultiSearch.from_response(completion)


if __name__ == "__main__":
    queries = segment(
        "Please send me the video from last week about the investment case study and also documents about your GPDR policy?"
    )

    queries.execute()
    # >>> Searching for `Video` with query `investment case study` using `SearchType.VIDEO`
    # >>> Searching for `Documents` with query `GPDR policy` using `SearchType.EMAIL`
