# Using the pipeline

The pipeapi is some syntactic sugar to help build prompts in a readable way that avoids having to remember best practices around wording and structure. Examples include adding tips, tagging data with xml, or even including the chain of thought prompt as an assistant message.

### Example Pipeline

```python
from openai_function_call import OpenAISchema, dsl
from pydantic import Field


class SearchQuery(OpenAISchema):
    query: str = Field(
        ...,
        description="Detailed, comprehensive, and specific query to be used for semantic search",
    )


SearchResponse = dsl.MultiTask(
    subtask_class=SearchQuery,
)


task = (
    dsl.ChatCompletion(name="Segmenting Search requests example")
    | dsl.SystemTask(task="Segment search results")
    | dsl.TaggedMessage(
        content="can you send me the data about the video investment and the one about spot the dog?",
        tag="query",
    )
    | dsl.TipsMessage(
        tips=[
            "Expand query to contain multiple forms of the same word (SSO -> Single Sign On)",
            "Use the title to explain what the query should return, but use the query to complete the search",
            "The query should be detailed, specific, and cast a wide net when possible",
        ]
    )
    | SearchResponse
)
search_request = task.create()  # type: ignore
assert isinstance(search_request, SearchResponse)
print(search_request.json(indent=2))
```

Output

```json
{
  "tasks": [
    {
      "query": "data about video investment"
    },
    {
      "query": "data about spot the dog"
    }
  ]
}
```

## Inspecting the API Call

To make it easy for you to understand what this api is doing we default only construct the kwargs for the chat completion call.

```python
print(task.kwargs)
```

```json
{
 "messages": [
  {
   "role": "system",
   "content": "You are a world class state of the art algorithm capable of correctly completing the following task: `Segment search results`."
  },
  {
   "role": "user",
   "content": "Consider the following data:\n\n<query>can you send me the data about the video investment and the one about spot the dog?</query>"
  },
  {
   "role": "user",
   "content": "Here are some tips to help you complete the task:\n\n* Expand query to contain multiple forms of the same word (SSO -> Single Sign On)\n* Use the title to explain what the query should return, but use the query to complete the search\n* The query should be detailed, specific, and cast a wide net when possible"
  }
 ],
 "functions": [
  {
   "name": "MultiSearchQuery",
   "description": "Correctly segmented set of search queries",
   "parameters": {
    "type": "object",
    "properties": {
     "tasks": {
      "description": "Correctly segmented list of `SearchQuery` tasks",
      "type": "array",
      "items": {
       "$ref": "#/definitions/SearchQuery"
      }
     }
    },
    "definitions": {
     "SearchQuery": {
      "type": "object",
      "properties": {
       "query": {
        "description": "Detailed, comprehensive, and specific query to be used for semantic search",
        "type": "string"
       }
      },
      "required": [
       "query"
      ]
     }
    },
    "required": [
     "tasks"
    ]
   }
  }
 ],
 "function_call": {
  "name": "MultiSearchQuery"
 },
 "max_tokens": 1000,
 "temperature": 0.1,
 "model": "gpt-3.5-turbo-0613"
}
```
