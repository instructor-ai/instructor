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

import enum
from typing import List

import openai
from pydantic import Field

from openai_function_call import OpenAISchema


class SearchType(str, enum.Enum):
    """Enumeration representing the types of searches that can be performed."""

    VIDEO = "video"
    EMAIL = "email"


class Search(OpenAISchema):
    """
    Class representing a single search query which contains title, query and the search type
    """

    search_title: str = Field(..., description="Title of the request")
    query: str = Field(..., description="Query to search for relevant content")
    type: SearchType = Field(..., description="Type of search")

    async def execute(self):
        import asyncio

        await asyncio.sleep(1)
        print(
            f"Searching for `{self.search_title}` with query `{self.query}` using `{self.type}`"
        )


class MultiSearch(OpenAISchema):
    """
    Class representing multiple search queries.
    Make sure they contain all the required attributes

    Args:
        searches (List[Search]): The list of searches to perform.
    """

    searches: List[Search] = Field(..., description="List of searches")

    def execute(self):
        import asyncio

        loop = asyncio.get_event_loop()

        tasks = asyncio.gather(*[search.execute() for search in self.searches])
        return loop.run_until_complete(tasks)


def segment(data: str) -> MultiSearch:
    """
    Convert a string into multiple search queries using OpenAI's GPT-3 model.

    Args:
        data (str): The string to convert into search queries.

    Returns:
        MultiSearch: An object representing the multiple search queries.
    """

    completion = openai.ChatCompletion.create(
        model="gpt-4-0613",
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
