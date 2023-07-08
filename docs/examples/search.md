# Example: Segmenting search queries 

This example will try to highlight a few ways of leveraging `MultiTask`, `enum.Enum`, and using methods to create powerful extrations that make using LLMS feel like regular code. 

## Defining the structures 

Lets model the problem as breaking down a search request into a list of search requests, we'll add some enums to make it interesting and 
take advantage of the fact that these are python objects and add some additional query logic

```python
import enum

from pydantic import Field
from openai_function_call import OpenAISchema


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
        print(
            f"Searching for `{self.title}` with query `{self.query}` using `{self.type}`"
        )
```

!!! tip "Data can have computation!"
    Notice that we can have an `execute` method on the class that routes the search query based on the enum type.

    ```python
    async def execute(self)
        if self.type == SearchType.VIDEO:
            ...
        else:
            ...
        return 
    ```

    This can be called after to run the queries

### Multiple queries

Often times a request might have multiple queries, we can manually create another class with a list attribute to represent this

```python
class MultiSearch(OpenAISchema):
    "Correctly segmented set of search results"
    tasks: List[Search]
```

!!! tips "Prompting is important"
    Its important to add docstrings and field descriptions to improve your prompting, even adding 'correctly' often leads to better results.

!!! usage "Multiple Tasks"
    The pattern of defining a task and then multiple tasks is common enought that I made a helper `openai_function_call.dsl.MultiTask` to avoid writing generic code.

## Putting it all together

Without using the lets define a function with some type hints 

```python
def segment(data: str) -> MultiSearch:
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        temperature=0.1,
        functions=[MultiSearch.openai_schema],
        function_call={"name": MultiSearch.openai_schema["name"]},
        messages=[
            {
                "role": "user",
                "content": f"Consider the data below: '\n{data}' and segment it into multiple search queries",
            },
        ],
        max_tokens=1000,
    )
    return MultiSearch.from_response(completion)
```

!!! tips "Typehints"
    If you're using an IDE its a great idea to have type hints as
    they make your developer experience better. Its easier to read, and intelligent autocomplete gives you more confidence.

## Evaluating an example

```python
queries = segment(
    "Please send me the video from last week about the investment case study and also documents about your GPDR policy?"
)
asyncio.gather([q.execute() for q in queries.tasks])
```

By using async we can execute the queries efficiently with fairly modular and simple code.

```bash
Searching for `Video` with query `investment case study` using `SearchType.VIDEO`
Searching for `Documents` with query `GPDR policy` using `SearchType.EMAIL`
```