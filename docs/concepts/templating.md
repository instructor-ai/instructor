# Templating

Instructor uses Jinja to provide templating support for prompts - this allows us to separate our prompt from the data we pass in, make prompts easier to manage and res-use and most importantly, have rendering and validation logic expressed within the prompt itself.

## Using `context`

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
        {"role": "user", "content": """Extract the information from the
        following text: {{ name }} is {{ age }} years old""" # (1)!
        },
    ],
    response_model=User,
    context = { # (2)!
        "name": "John Doe",
        "age": 30,
    }
)

print(resp)
#> User(name='John Doe', age=30)
```

1. Declare jinja template variables inside the prompt itself (e.g. `{{ name }}`)
2. Pass in the variables to be used in the `context` parameter

### Integration with Pydantic

We can use the `context` with our normal Pydantic validators.

```python hl_lines="15-16 26-30"
import openai
import instructor
from pydantic import BaseModel, ValidationInfo, field_validator

client = instructor.from_openai(openai.OpenAI())

class User(BaseModel):
    name: str
    age: int

    @field_validator("name")
    def validate_name(cls, v:str, info:ValidationInfo)-> str:
        context = info.context

        if context["awesome_people"] and v in context["awesome_people"]: # (1)!
            raise ValueError(f"{context['awesome_people']} should have their names fully capitalized with a star emoji. Make the edits")

        return v.upper()

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Extract the information from the following text: {{ name }} is {{ age }} years old"},
    ],
    response_model=User,
    context = { # (2)!
        "name": "Chris",
        "age": 27,
        "awesome_people": ["Chris"]
    }
)

print(resp)
# name='CHRIS ⭐' age=27
```

1. Access the variables passed into the `context` variable inside your Pydantic validator

2. Pass in the variables to be used for validation and/or rendering into the `context` parameter

### Jinja Syntax

Because we're using jinja to render the prompts, we can use all of the familiar jinja syntax that you're used to. This makes it easy to render list of items, conditionals and more. This is incredibly useful because we can even call specific functions and methods within Jinja itself!

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
            """
        },
    ],
    response_model=Response,
    context = {
        "role": "professional educator",
        "question": "What is the capital of France?",
        "context": [
            {"id": 1, "text": "Paris is the capital of France."},
            {"id": 2, "text": "France is a country in Europe."}
        ],
        "rules": ["Use markdown."]
    }
)

print(resp)
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
            "content": "{{ user.name }} is `{{ user.address.get_secret_value() }}`, normalize it to an address object"
        },
    ],
    context={"user": context},
    response_model=Address,
)
print(context)
# > UserContext(username='jliu', address="******")
print(address)
# > Address(street='******', city="Toronto", state="Ontario", zipcode="M5A 0J3")
```

This allows you to preserve your sensitive information while still using it in your prompts.

## Different Clients

Templating support currently works for OpenAI, Anthropic, Gemini and VertexAI clients. See below for examples for each of these clients.

### OpenAI

### Anthropic

### Gemini

### VertexAI
