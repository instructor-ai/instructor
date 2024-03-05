# Structured Outputs with Groq AI

If you want to try this example using `instructor hub`, you can pull it by running

```bash
instructor hub pull --slug groq --py > groq_example.py
```

you'll need to sign up for an account and get an API key. You can do that [here](https://console.groq.com/docs/quickstart).

```bash
export GROQ_API_KEY=<your-api-key-here>
pip install groq
```

!!! note "Other Languages"

    This blog post is written in Python, but the concepts are applicable to other languages as well, as we currently have support for [Javascript](https://instructor-ai.github.io/instructor-js), [Elixir](https://hexdocs.pm/instructor/Instructor.html) and [PHP](https://github.com/cognesy/instructor-php/).

<!-- more -->

## Patching

Instructor's patch enhances the openai api it with the following features:

- `response_model` in `create` calls that returns a pydantic model
- `max_retries` in `create` calls that retries the call if it fails by using a backoff strategy

!!! note "Learn More"

    To learn more, please refer to the [docs](../index.md). To understand the benefits of using Pydantic with Instructor, visit the tips and tricks section of the [why use Pydantic](../why.md) page.

## Groq AI

While Groq AI does not support function calling directly, you can still leverage the MD_JSON mode for structured outputs.

!!! note "Getting access"

    If you want to try this out for yourself check out the [docs](https://console.groq.com/docs/quickstart)


```python
import os
import instructor

from groq import Groq
from pydantic import BaseModel

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# By default, the patch function will patch the ChatCompletion.create and ChatCompletion.create methods to support the response_model parameter
client = instructor.patch(
    client, mode=instructor.Mode.MD_JSON
)

# Now, we can use the response_model parameter using only a base model
# rather than having to use the OpenAISchema class
class UserExtract(BaseModel):
    name: str
    age: int


user: UserExtract = client.chat.completions.create(
    model="mixtral-8x7b-32768",
    response_model=UserExtract,
    messages=[
        {"role": "user", "content": "Extract jason is 25 years old"},
    ],
)

assert isinstance(user, UserExtract), "Should be instance of UserExtract"
assert user.name.lower() == "jason"
assert user.age == 25

print(user.model_dump_json(indent=2))
"""
{
  "name": "jason",
  "age": 25
}
"""
```