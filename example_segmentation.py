from openai_function_call import OpenAISchema
from pydantic import Field
from typing import List
from tenacity import retry, stop_after_attempt
import openai


class Search(OpenAISchema):
    """
    Search query for a single request

    Tips:
    - Be specific with your query, use key words and multiple representations of the same thing, e.g. "video" and "video clip" or "SSO" and "single sign on"
    - Use the title to describe the request, e.g. "Video from last week about the investment case study"
    """

    title: str = Field(..., description="Title of the request")
    query: str = Field(..., description="Query to search for relevant content")

    async def execute(self):
        import asyncio

        await asyncio.sleep(1)
        print(f"Searching for `{self.title}` with query `{self.query}`")


class MultiSearch(OpenAISchema):
    """
    Segment a request into multiple search queries

    Tips:
    - Do not overlap queries, e.g. "video" and "video clip" are too similar
    """

    searches: List[Search] = Field(..., description="List of searches")

    def execute(self):
        import asyncio

        loop = asyncio.get_event_loop()

        tasks = asyncio.gather(*[search.execute() for search in self.searches])
        return loop.run_until_complete(tasks)


@retry(stop=stop_after_attempt(3))
def segment(data: str) -> MultiSearch:
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        temperature=0,
        functions=[MultiSearch.openai_schema],
        messages=[
            {
                "role": "system",
                "content": "You must use the tool given to response.",
            },
            {
                "role": "user",
                "content": f"Consider the data below:\n{data} and segment it into multiple search queries. You must use `MultiStep` to do this.",
            },
        ],
        max_tokens=1000,
    )
    return MultiSearch.from_response(completion)


if __name__ == "__main__":
    queries = segment(
        "Please send me the video from last week about the investment case study and also documents about your GPDR policy?"
    )

    results = queries.execute()
