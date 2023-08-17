# Example: Segmenting Search Queries

In this example, we will demonstrate how to leverage the `MultiTask` and `enum.Enum` features of OpenAI Function Call to segment search queries. We will define the necessary structures using Pydantic and demonstrate how segment queries into multiple sub queries and execute them in parallel with `asyncio`.

!!! tips "Motivation"

    Extracting a list of tasks from text is a common use case for leveraging language models. This pattern can be applied to various applications, such as virtual assistants like Siri or Alexa, where understanding user intent and breaking down requests into actionable tasks is crucial. In this example, we will demonstrate how to use OpenAI Function Call to segment search queries and execute them in parallel.


## Defining the Structures

Let's model the problem as breaking down a search request into a list of search queries. We will use an enum to represent different types of searches and take advantage of Python objects to add additional query logic.

```python
import enum
from pydantic import Field
from instructor import OpenAISchema

class SearchType(str, enum.Enum):
    """Enumeration representing the types of searches that can be performed."""
    VIDEO = "video"
    EMAIL = "email"

class Search(OpenAISchema):
    """
    Class representing a single search query.
    """
    title: str = Field(..., description="Title of the request")
    query: str = Field(..., description="Query to search for relevant content")
    type: SearchType = Field(..., description="Type of search")

    async def execute(self):
        print(f"Searching for `{self.title}` with query `{self.query}` using `{self.type}`")
```

Notice that we have added the `execute` method to the `Search` class. This method can be used to route the search query based on the enum type. You can add logic specific to each search type in the `execute` method.

Next, let's define a class to represent multiple search queries.

```python
from typing import List

class MultiSearch(OpenAISchema):
    "Correctly segmented set of search results"
    tasks: List[Search]
```

The `MultiSearch` class has a single attribute, `tasks`, which is a list of `Search` objects.

This pattern is so common that we've added a helper function `MultiTask` to makes this simpler 

```python
from instructor.dsl import MultiTask

MultiSearch = MultiTask(Search)
```

## Calling Completions

To segment a search query, we will use the base openai api. We can define a function that takes a string and returns segmented search queries using the `MultiSearch` class.

```python hl_lines="7 8"
import openai

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

The `segment` function takes a string `data` and creates a completion. It prompts the model to segment the data into multiple search queries and returns the result as a `MultiSearch` object.

## Evaluating an Example

Let's evaluate an example by segmenting a search query and executing the segmented queries.

```python
import asyncio

queries = segment("Please send me the video from last week about the investment case study and also documents about your GDPR policy?")

async def execute_queries(queries: Multisearch):
    await asyncio.gather(*[q.execute() for q in queries.tasks])

loop = asyncio.get_event_loop()
loop.run_until_complete(execute_queries())
loop.close()
```

In this example, we use the `segment` function to segment the search query. We then use `asyncio` to asynchronously execute the queries using the `execute` method defined in the `Search` class.

The output will be:

```
Searching for `Please send me the video from last week about the investment case study` with query `Please send me the video from last week about the investment case study` using `SearchType.VIDEO`
Searching for `also documents about your GDPR policy?` with query `also documents about your GDPR policy?` using `SearchType.EMAIL`
```