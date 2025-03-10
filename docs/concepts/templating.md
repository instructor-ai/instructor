---
title: Effective Prompt Templating with Jinja and Pydantic
description: Learn to dynamically create prompts using Jinja templating and validate them with Pydantic for enhanced flexibility and security.
---

# Prompt Templating

With Instructor's Jinja templating, you can:

- Dynamically adapt prompts to any context
- Easily manage and version your prompts better
- Integrate seamlessly with validation processes
- Handle sensitive information securely

Our solution offers:

- Separation of prompt structure and content
- Complex logic implementation within prompts
- Template reusability across scenarios
- Enhanced prompt versioning and logging
- Pydantic integration for validation and type safety

## Context is available to the templating engine

The `context` parameter is a dictionary that is passed to the templating engine. It is used to pass in the relevant variables to the templating engine. This single `context` parameter will be passed to jinja to render out the final prompt.

```python hl_lines="14-15 19-22"
import openai
import instructor
from pydantic import BaseModel

client = instructor.from_openai(openai.OpenAI())


class User(BaseModel):
    name: str
    age: int


resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": """Extract the information from the
        following text: `{{ data }}`""",  # (1)!
        },
    ],
    response_model=User,
    context={"data": "John Doe is thirty years old"},  # (2)!
)

print(resp)
#> name='John Doe' age=30
```

1. Declare jinja style template variables inside the prompt itself (e.g. `{{ name }}`)
2. Pass in the variables to be used in the `context` parameter

### Context is available to Pydantic validators

In this example, we demonstrate how to leverage the `context` parameter with Pydantic validators to enhance our validation and data processing capabilities. By passing the `context` to the validators, we can implement dynamic validation rules and data transformations based on the input context. This approach allows for flexible and context-aware validation, such as checking for banned words or applying redaction patterns to sensitive information.

```python hl_lines="15-16 26-30"
import openai
import instructor
from pydantic import BaseModel, ValidationInfo, field_validator
import re

client = instructor.from_openai(openai.OpenAI())


class Response(BaseModel):
    text: str

    @field_validator('text')
    @classmethod
    def redact_regex(cls, v: str, info: ValidationInfo):
        context = info.context
        if context:
            redact_patterns = context.get('redact_patterns', [])
            for pattern in redact_patterns:
                v = re.sub(pattern, '****', v)
        return v


response = client.create(
    model="gpt-4o",
    response_model=Response,
    messages=[
        {
            "role": "user",
            "content": """
                Write about a {{ topic }}

                {% if banned_words %}
                You must not use the following banned words:

                <banned_words>
                {% for word in banned_words %}
                * {{ word }}
                {% endfor %}
                </banned_words>
                {% endif %}
              """,
        },
    ],
    context={
        "topic": "jason and now his phone number is 123-456-7890",
        "redact_patterns": [
            r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",  # Phone number pattern
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN pattern
        ],
    },
    max_retries=3,
)

print(response.text)
"""
Jason is a vibrant and dynamic individual known for his charismatic personality and entrepreneurial spirit. He has always been passionate about technology and innovation, which led him to start his own tech company. Throughout his career, Jason has been dedicated to making a significant impact in the tech industry, always seeking out new opportunities to learn and grow.

In addition to his professional endeavors, Jason is an adventurous person who loves to travel and explore new places. He finds joy in experiencing different cultures and meeting new people, which has contributed to his broad worldview and understanding of global markets.

Jason’s journey is one of hard work, resilience, and determination, as he continuously pushes the boundaries to achieve his goals and inspire those around him.

(Note: Personal phone numbers should remain confidential and not be shared publicly to protect privacy.)
"""
```

1. Access the variables passed into the `context` variable inside your Pydantic validator

2. Pass in the variables to be used for validation and/or rendering into the `context` parameter

### Jinja Syntax

Jinja is used to render the prompts, allowing the use of familiar Jinja syntax. This enables rendering of lists, conditionals, and more. It also allows calling functions and methods within Jinja.

This makes formatting of prompts and rendering logic extremely easy.

```python hl_lines="29-34 37-43"
import openai
import instructor
from pydantic import BaseModel

client = instructor.from_openai(openai.OpenAI())


class Citation(BaseModel):
    source_ids: list[int]
    text: str


class Response(BaseModel):
    answer: list[Citation]


resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": """
                You are a {{ role }} tasks with the following question

                <question>
                {{ question }}
                </question>

                Use the following context to answer the question, make sure to return [id] for every citation:

                <context>
                {% for chunk in context %}
                  <context_chunk>
                    <id>{{ chunk.id }}</id>
                    <text>{{ chunk.text }}</text>
                  </context_chunk>
                {% endfor %}
                </context>

                {% if rules %}
                Make sure to follow these rules:

                {% for rule in rules %}
                  * {{ rule }}
                {% endfor %}
                {% endif %}
            """,
        },
    ],
    response_model=Response,
    context={
        "role": "professional educator",
        "question": "What is the capital of France?",
        "context": [
            {"id": 1, "text": "Paris is the capital of France."},
            {"id": 2, "text": "France is a country in Europe."},
        ],
        "rules": ["Use markdown."],
    },
)

print(resp)
#> answer=[Citation(source_ids=[1], text='The capital of France is Paris.')]
# answer=[Citation(source_ids=[1], text='The capital of France is Paris.')]
```

### Working with Secrets

Your prompts might need to include sensitive user information when they're sent to your model provider. This is probably something you don't want to hard code into your prompt or captured in your logs. An easy way to get around this is to use the `SecretStr` type from `Pydantic` in your model definitions.

```python
from pydantic import BaseModel, SecretStr
import instructor
import openai


class UserContext(BaseModel):
    name: str
    address: SecretStr


class Address(BaseModel):
    street: SecretStr
    city: str
    state: str
    zipcode: str


client = instructor.from_openai(openai.OpenAI())
context = UserContext(name="scolvin", address="secret address")

address = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": "{{ user.name }} is `{{ user.address.get_secret_value() }}`, normalize it to an address object",
        },
    ],
    context={"user": context},
    response_model=Address,
)
print(context)
#> name='scolvin' address=SecretStr('**********')
print(address)
#> street=SecretStr('**********') city='scolvin' state='NA' zipcode='00000'
```

This allows you to preserve your sensitive information while still using it in your prompts.

## Security

We use the `jinja2.sandbox.SandboxedEnvironment` to prevent security issues with the templating engine. This means that you can't use arbitrary python code in your prompts. But this doesn't mean that you should pass untrusted input to the templating engine, as this could still be abused for things like Denial of Service attacks.

You should [always sanitize](https://jinja.palletsprojects.com/en/stable/sandbox/#security-considerations) any input that you pass to the templating engine.
