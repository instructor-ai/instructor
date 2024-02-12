# Patching the OpenAI Client

Instructor enhances client functionality with three new keywords for backwards compatibility. This allows use of the enhanced client as usual, with structured output benefits.

- `response_model`: Defines the response type for `chat.completions.create`.
- `max_retries`: Determines retry attempts for failed `chat.completions.create` validations.
- `validation_context`: Provides extra context to the validation process.

There are two ways to patch the OpenAI client: patching the client itself, or patching a specific function. The former is more general, while the latter is more specific.

Then there are a handful of modes to choose from:

1. **Function Calling**: The primary method. Use this for stability and testing.
2. **Tool Calling**: Useful in specific scenarios; lacks the reasking feature of OpenAI's tool calling API.
3. **JSON Mode**: Offers closer adherence to JSON but with more potential validation errors. Suitable for specific non-function calling clients.
4. **Markdown JSON Mode**: Experimental, not recommended.
5. **JSON Schema Mode**: Only available with the [Together](../blog/posts/together.md) and [Anyscale](../blog/posts/anyscale.md) patches.

## Patching Modes

We support two kinds of patches One that patches anything isomorphic to the OpenAI client and the other to a specific create call

### Patch Client

```python
import instructor
from openai import OpenAI

client = instructor.patch(OpenAI())
```

### Patch Create

This allows you to patch any function that is isomorphic to `chat.completions.create`. function with messages and other parameters.

```python
import instructor
from openai import OpenAI

create_fn = OpenAI().chat.completions.create
create = instructor.patch(create=create_fn)
```

## Patching Modes

### Function Calling

```python
import instructor
from openai import OpenAI

client = instructor.patch(OpenAI(), mode=instructor.Mode.FUNCTIONS)
```

### Tool Calling

```python
import instructor
from openai import OpenAI

client = instructor.patch(OpenAI(), mode=instructor.Mode.TOOLS)
```

### JSON Mode

```python
import instructor
from instructor import Mode
from openai import OpenAI

client = instructor.patch(OpenAI(), mode=Mode.JSON)
```

### Markdown JSON Mode

!!! warning "Experimental"

    This is not recommended, and may not be supported in the future, this is just left to support vision models.

```python
import instructor
from openai import OpenAI

client = instructor.patch(OpenAI(), mode=instructor.Mode.MD_JSON)
```

### JSON Schema Mode

This is only available with the [Together](../blog/posts/together.md) and [Anyscale](../blog/posts/anyscale.md) patches.

```python
import instructor
from openai import OpenAI

client = instructor.patch(OpenAI(
    ...
), mode=instructor.Mode.JSON_SCHEMA)
```
