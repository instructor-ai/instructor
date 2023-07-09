# Using the ChatCompletion pipeline

The pipeline api is just syntactic sugar to help build prompts in a readable way that avoids having to remember best practices around wording and structure. Examples include adding tips, tagging data with xml, or even including the chain of thought prompt as an assistant message.

## Example Pipeline

Here we'll define a task to segment queries and add some more instructions via the prompt pipeline api.

### Designing the schema

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
```

!!! tip "MultiTask"
    To learn more about what multi task does, checkout the [MultiTask](multitask.md) documentation


### Building our prompts

We dont deal with prompt templates and treat chat, message, output schema as first class citizens and then pipe them into a completion object.

!!! note "Whats that?"
    The pipe `|` is an overloaded operator that lets us cleanly compose our prompts.

    `ChatCompletion` contains all the configuration for the model while we use `|` to build our prompt

    We can then chain `|` together to add `Messages` or `OpenAISchema` and `ChatCompletion` will build out query for us while giving us a readable block to code to look ad

    To see what 'message templates' are available check out our [docs](chat-completion.md)

```python
task = (
    dsl.ChatCompletion(
        name="Segmenting Search requests example",
        model='gpt-3.5-turbo-0613,
        max_token=1000)
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

!!! tip
    If you want to see what its actually sent to OpenAI scroll to the bottom of the page! 

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
