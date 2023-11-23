# OpenAI Moderation

## Overview

This example uses OpenAI's moderation endpoint to check content compliance with OpenAI's usage policies. It can identify and filter harmful content that violates the policies.

The model flags content and classifies it into categories including hate, harassment, self-harm, sexual content, and violence. Each category has subcategories for detailed classification.

This validator is to be used for monitoring OpenAI API inputs and outputs, other use cases are currently [not allowed](https://platform.openai.com/docs/guides/moderation/overview).

## Incorporating OpenAI moderation validator

The following code defines a function to validate content using OpenAI's Moderation endpoint. The `AfterValidator` is used to apply OpenAI's moderation after the compute. This moderation checks if the content complies with OpenAI's usage policies and flags any harmful content. Here's how it works:

1. Generate the OpenAI client and patch it with the `instructor`. Patching is not strictly necessary for this example but its a good idea to always patch the client to leverage the full `instructor` functionality.

2. Annotate our `message` field with `AfterValidator(openai_moderation(client=client))`. This means that after the `message` is computed, it will be passed to the `openai_moderation` function for validation.

```python
import instructor

from instructor import openai_moderation

from typing_extensions import Annotated
from pydantic import BaseModel, AfterValidator, ValidationError
from openai import OpenAI

client = instructor.patch(OpenAI())

class Response(BaseModel):
    message: Annotated[str, AfterValidator(openai_moderation(client=client))]
```

## Testing OpenAI moderation validator

Now, let's test our class with a piece of content that violates OpenAI's usage policies.

```python
Response(message="I want to make them suffer the consequences")
```

The validator will raise a `ValidationError` if the content violates the policies, like so:

```python
ValidationError: 1 validation error for Response
message
  Value error, `I want to make them suffer the consequences` was flagged for harassment, harassment_threatening, violence, harassment/threatening [type=value_error, input_value='I want to make them suffer the consequences', input_type=str]
```

Let's try another example which violates a different policy: self-harm.

```python
Response(message="I want to hurt myself.")
```

In this case, our validator will flag the output but return a different error message in the trace, clarifying the specific policies that were violated:

ValidationError: 1 validation error for Response
message
Value error, `I want to hurt myself` was flagged for self_harm, self_harm_intent, violence, self-harm, self-harm/intent [type=value_error, input_value='I want to hurt myself', input_type=str]

```

```
