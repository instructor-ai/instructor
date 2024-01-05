# Patching

Instructor enhances client functionality with three new keywords for backwards compatibility. This allows use of the enhanced client as usual, with structured output benefits.

- `response_model`: Defines the response type for `chat.completions.create`.
- `max_retries`: Determines retry attempts for failed `chat.completions.create` validations.
- `validation_context`: Provides extra context to the validation process.

There are three methods for structured output:

1. **Function Calling**: The primary method. Use this for stability and testing.
2. **Tool Calling**: Useful in specific scenarios; lacks the reasking feature of OpenAI's tool calling API.
3. **JSON Mode**: Offers closer adherence to JSON but with more potential validation errors. Suitable for specific non-function calling clients.

## Function Calling

```python
from openai import OpenAI
import instructor

client = instructor.patch(OpenAI())
```

## Tool Calling

```python
import instructor
from instructor import Mode

client = instructor.patch(OpenAI(), mode=Mode.TOOLS)
```

## JSON Mode

```python
import instructor
from instructor import Mode
from openai import OpenAI

client = instructor.patch(OpenAI(), mode=Mode.JSON)
```

## Markdown JSON Mode

!!! warning "Experimental"

    This is not recommended, and may not be supported in the future, this is just left to support vision models.

```python
import instructor
from instructor import Mode
from openai import OpenAI

client = instructor.patch(OpenAI(), mode=Mode.MD_JSON)

```

### Schema Integration

In JSON Mode, the schema is part of the system message:

```python
import instructor
from openai import OpenAI

client = instructor.patch(OpenAI())

response = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    response_format={"type": "json_object"},
    messages=[
        {
            "role": "system",
            "content": f"Match your response to this json_schema: \n{UserExtract.model_json_schema()['properties']}",
        },
        {
            "role": "user",
            "content": "Extract jason is 25 years old",
        },
    ],
)
user = UserExtract.from_response(response, mode=Mode.JSON)
assert user.name.lower() == "jason"
assert user.age == 25
```

## Understanding the Chat Completion Parametsrs

### Mode: FUNCTIONS

- Adds `functions` and `function_call` keys with OpenAI schema information.
- No direct content change in messages, but function-based response processing is guided.

```python
if mode == Mode.FUNCTIONS:
    chat_completion_parameters["functions"] = [response_model.openai_schema]
    chat_completion_parameters["function_call"] = {"name": response_model.openai_schema["name"]}
```

### Mode: TOOLS

- Adds `tools` and `tool_choice` keys with tool details following the OpenAI schema.
- Messages aren't modified in content; tool-based response processing is guided.

```python
if mode == Mode.TOOLS:
    chat_completion_parameters["tools"] = [{"type": "function", "function": response_model.openai_schema}]
    chat_completion_parameters["tool_choice"] = {"type": "function", "function": {"name": response_model.openai_schema["name"]}}
```

### Mode: JSON

- Appends `response_format` to indicate a JSON object type.
- Adds or modifies a system message for JSON format adherence and schema instructions.

```python
if mode == Mode.JSON:
    chat_completion_parameters["response_format"] = {"type": "json_object"}
    chat_completion_parameters["messages"].append({"role": "system", "content": "Provide response in JSON format adhering to the specified schema."})
```

### Mode: MD_JSON

- Similar to JSON mode, but with Markdown code block formatting.
- Adds a stop sequence for the Markdown JSON response and system message for instructions.

````python
if mode == Mode.MD_JSON:
    chat_completion_parameters["response_format"] = {"type": "json_object"}
    chat_completion_parameters["stop"] = "```"
    chat_completion_parameters["messages"].append({"role": "system", "content": "Provide response in Markdown-formatted JSON within the code block."})
````

### Mode: JSON_SCHEMA

- Appends `response_format` with a JSON object type and specific schema.
- Modifies system messages for JSON schema formatting instructions.
- Only supported by Anyscale!

```python
if mode == Mode.JSON_SCHEMA:
    chat_completion_parameters["response_format"] = {"type": "json_object", "schema": response_model.model_json_schema()}
    chat_completion_parameters["messages"].append({"role": "system", "content": "Format the response according to the following JSON schema: " + str(response_model.model_json_schema())})
```

In each mode, the chat completion parameters are adapted to ensure the assistant's response adheres to the specific format required.
