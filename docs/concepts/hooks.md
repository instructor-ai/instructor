# Hooks

Hooks provide a powerful mechanism for intercepting and handling events during the completion and parsing process in the Instructor library. They allow you to add custom behavior, logging, or error handling at various stages of the API interaction.

## Overview

The Hooks system in Instructor is based on the `Hooks` class, which manages event registration and emission. It supports several predefined events that correspond to different stages of the completion and parsing process.

## Supported Hook Events

### `completion:kwargs`

This hook is emitted when completion arguments are provided. It receives all arguments passed to the completion function. These will contain the `model`, `messages`, `tools`, AFTER any `response_model` or `validation_context` parameters have been converted to their respective values.

```python
def handler(*args: Any, **kwargs: Any) -> None: ...
```

### `completion:response`

This hook is emitted when a completion response is received. It receives the raw response object from the completion API.

```python
def handler(response: Any) -> None: ...
```

### `completion:error`

This hook is emitted when an error occurs during completion before any retries are attempted and the response is parsed as a pydantic model.

```python
def handler(error: Exception) -> None: ...
```

### `parse:error`

This hook is emitted when an error occurs during parsing of the response as a pydantic model. This can happen if the response is not valid or if the pydantic model is not compatible with the response.

```python
def handler(error: Exception) -> None: ...
``` 

### `completion:last_attempt`

This hook is emitted when the last retry attempt is made.

```python
def handler() -> None: ...
```

### Registering Hooks

You can register hooks using the `on` method of the Instructor client or a `Hooks` instance. Here's an example:

```python
import instructor
import openai
import pprint

client = instructor.from_openai(openai.OpenAI())


def log_completion_kwargs(*args, **kwargs):
    pprint.pprint({"args": args, "kwargs": kwargs})


client.on("completion:kwargs", log_completion_kwargs)

resp = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello, world!"}],
    response_model=str,
)
print(resp)
```

### Emitting Events

Events are automatically emitted by the Instructor library at appropriate times. You don't need to manually emit events in most cases.

### Removing Hooks

You can remove a specific hook using the `off` method:

```python
client.off("completion:kwargs", log_completion_kwargs)
```

### Clearing Hooks

To remove all hooks for a specific event or all events:

```python
# Clear hooks for a specific event
client.clear("completion:kwargs")

# Clear all hooks
client.clear()
```

## Example: Logging and Debugging

Here's a comprehensive example demonstrating how to use hooks for logging and debugging:

```python
import instructor
import openai
import pydantic
from pydantic import field_validator
import pprint


def log_completion_kwargs(*args: Any, **kwargs: Any) -> None:
    """Log the completion kwargs."""
    print("## Completion kwargs:")
    pprint.pprint({"args": args, "kwargs": kwargs})


def log_completion_response(response: Any) -> None:
    """Log the completion response."""
    print("## Completion response:")
    pprint.pprint(response.model_dump())


def log_completion_error(error: Exception) -> None:
    """Log the completion error."""
    print("## Completion error:")
    pprint.pprint({"error": error})


def log_parse_error(error: Exception) -> None:
    """Log the parse error."""
    print("## Parse error:")
    pprint.pprint({"error": error})


# Create an Instructor client
client = instructor.patch(openai.OpenAI())

# Register hooks
client.on("completion:kwargs", log_completion_kwargs)
client.on("completion:response", log_completion_response)
client.on("completion:error", log_completion_error)
client.on("parse:error", log_parse_error)


# Define a model with a validator
class User(pydantic.BaseModel):
    name: str
    age: int

    @field_validator("age")
    def check_age(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Age cannot be negative")
        return v


# Use the client to create a completion
user = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "user",
            "content": "Extract the user name and age from the following text: 'John is -1 years old'",
        }
    ],
    response_model=User,
)

print(user)
```

This example demonstrates:

1. Defining hook handlers for different events.
2. Registering the hooks with the Instructor client.
3. Using a Pydantic model with a validator.
4. Making a completion request that will trigger various hooks.

The hooks will log information at different stages of the process, helping with debugging and understanding the flow of data.

## Best Practices

1. **Error Handling**: Always include error handling in your hook handlers to prevent exceptions from breaking the main execution flow. We will automatically warn if an exception is raised in a hook handler.

2. **Performance**: Keep hook handlers lightweight to avoid impacting the performance of the main application.

3. **Modularity**: Use hooks to separate concerns. For example, use hooks for logging, monitoring, or custom business logic without cluttering the main code.

4. **Consistency**: Use the same naming conventions and patterns across all your hooks for better maintainability.

5. **Documentation**: Document the purpose and expected input/output of each hook handler for easier collaboration and maintenance.

By leveraging hooks effectively, you can create more flexible, debuggable, and maintainable applications when working with the Instructor library and language models.
