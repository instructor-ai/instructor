"""
This example uses a nested object of Search and MultiSearch and 
supports retries with tenacity.
"""

from openai_function_call import OpenAISchema
from pydantic import Field
from typing import List
from tenacity import retry, stop_after_attempt
import openai


class Search(OpenAISchema):
    "Search query for a request,"

    title: str = Field(..., description="Title of the request")
    query: str = Field(..., description="Query to search for relevant content")


class MultiSearch(OpenAISchema):
    "Search query for multiple requests"
    searches: List[Search] = Field(..., description="List of searches")


@retry(stop=stop_after_attempt(3))
def segment(data: str) -> MultiSearch:
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        temperature=0,
        functions=[MultiSearch.openai_schema],
        messages=[
            {
                "role": "system",
                "content": "You must use the `MultiSearch` tool to search for multiple requests at once. You must response with a list of search queries",
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
    for query in queries.searches:
        print(query)
        
    # title='Video about investment case study from last week' query='investment case study video last week'
    # title='Documents about GPDR policy' query='GPDR policy documents'
