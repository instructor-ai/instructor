# Writing prompts with `ChatCompletion`

The ChatCompletion pipeline API provides a convenient way to build prompts with clear instructions and structure. It helps avoid the need to remember best practices for wording and prompt construction. This documentation will demonstrate an example pipeline and guide you through the process of using it.

Our goals are to:

1. Define some best practices with a light abstraction over a chat message
2. Allow the pipeline to be intuitive and readable.
3. Abstract the output shape and deserialization to better usability

## Example Pipeline

We will begin by defining a task to segment queries and add instructions using the prompt pipeline API.

1. We want to define a search object to extract
2. We want to extract multiple instances of such an object
3. We want to define the pipeline with a set of instructions
4. We want to easily call OpenAI and extract the data back out of the competion

!!! note "Applications"
    Extracted a repeated task out of instructions is a fairly common task.
    Prompting tips have been to define the task clearly, model the output object and provide tips to the llm for better performance. Something like this can be used to power agents like Siri or Alexa in performing multiple tasks in one request. [Read more](examples/search.md)

### Designing the Schema

First, let's design the schema for our task. In this example, we will have a `SearchQuery` schema with a single field called `query`. The `query` field will represent a detailed, comprehensive, and specific query to be used for semantic search.

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
    To learn more about the `MultiTask` functionality, you can refer to the [MultiTask](multitask.md) documentation.

### Building our Prompts

Next, let's write out prompt using the pipeline style. We will leverage the features provided by the `ChatCompletion` class and utilize the `|` operator to chain different components of our prompt together.

```python
task = (
    dsl.ChatCompletion(#(1)!
        name="Segmenting Search requests example",
        model='gpt-3.5-turbo-0613,
        max_token=1000) 
    | dsl.SystemTask(task="Segment search results") #(2)!
    | dsl.TaggedMessage(#(3)!
        content="can you send me the data about the video investment and the one about spot the dog?",
        tag="query") 
    | dsl.TipsMessage(#(4)!
        tips=[
            "Expand query to contain multiple forms of the same word (SSO -> Single Sign On)",
            "Use the title to explain what the query should return, but use the query to complete the search",
            "The query should be detailed, specific, and cast a wide net when possible",
        ]) 
    | SearchResponse #(5)!
)
```

1. Define the completion object (consider this both task and prompt)
2. SystemTask augments the `task` with "You are a *world class* ... *correctly* complete the task: {task}"
3. TaggedMessage wraps content with `<query></query>` to set clear boundaries for the data you wish to process
4. TipsMessages allows you to pass a list of strings as tips as a result we can potentially create this list dynamically
5. Last step defines the output model you want to use to parse the results if no outpout model is defined we revert to the usual openai completion.

The `ChatCompletion` class is responsible for model configuration, while the `|` operator allows us to construct the prompt in a readable manner. We can add `Messages` or `OpenAISchema` components to the prompt pipeline using `|`, and the `ChatCompletion` class will handle the prompt construction for us.

In the above example, we:

- Initialize a `ChatCompletion` object with the desired model and maximum token count.
- Add a `SystemTask` component to segment search results.
- Include a `TaggedMessage` component to provide a query with a specific tag.
- Use a `TipsMessage` component to include some helpful tips related to the task.
- Connect the `SearchResponse` schema to the pipeline.

Lastly, we create the `search_request` using `task.create()`. The `search_request` object will be of type `SearchResponse`, and we can print it as a JSON object.

!!! tip
    If you want to see the exact input sent to OpenAI, scroll to the bottom of the page.

```python
search_request = task.create()  # type: ignore
assert isinstance(search_request, SearchResponse)
print(search_request.json(indent=2))
```

The output will be a JSON object containing the segmented search queries.

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
