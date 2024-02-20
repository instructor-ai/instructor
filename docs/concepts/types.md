# Support for Simple Types

Aside from the recommended `pydantic.BaseModel`, and [Iterable](lists.md), and [Partial](partial.md),

Instructor supports simple types like `str`, `int`, `float`, `bool`, `Union`, `Literal`, out of the box. You can use these types directly in your response models.

To add more descriptions you can also use `typing.Annotated` to include more information about the type.

## Primatives

```python
import instructor
import openai

client = instructor.patch(openai.OpenAI())

# Response model with simple types like str, int, float, bool
resp = client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=bool,
    messages=[
        {
            "role": "user",
            "content": "Is it true that Paris is the capital of France?",
        },
    ],
)
assert resp is True, "Paris is the capital of France"
print(resp)
#> True
```

## Annotated

Annotations can be used to add more information about the type. This can be useful for adding descriptions to the type, along with more complex information like field names, and more.

```python
import instructor
import openai
from typing import Annotated
from pydantic import Field

client = instructor.patch(openai.OpenAI())

UpperCaseStr = Annotated[str, Field(description="string must be upper case")]

# Response model with simple types like str, int, float, bool
resp = client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=UpperCaseStr,
    messages=[
        {
            "role": "user",
            "content": "What is the capital of france?",
        },
    ],
)
assert resp == "PARIS", "Paris is the capital of France"
print(resp)
#> PARIS
```

## Literal

```python
import instructor
import openai
from typing import Literal

client = instructor.patch(openai.OpenAI())

resp = client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=Literal["BILLING", "SHIPPING"],
    messages=[
        {
            "role": "user",
            "content": "Classify the following messages: 'I am having trouble with my billing'",
        },
    ],
)
assert resp == "BILLING"
print(resp)
#> BILLING
```

## Literal

```python
import instructor
import openai
from enum import Enum


class Label(str, Enum):
    BILLING = "BILLING"
    SHIPPING = "SHIPPING"


client = instructor.patch(openai.OpenAI())

resp = client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=Label,
    messages=[
        {
            "role": "user",
            "content": "Classify the following messages: 'I am having trouble with my billing'",
        },
    ],
)
assert resp == Label.BILLING
print(resp)
#> BILLING
```

## List

```python
import instructor
import openai
from typing import List

client = instructor.patch(openai.OpenAI())

resp = client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=List[int],
    messages=[
        {
            "role": "user",
            "content": "Give me the first 5 prime numbers",
        },
    ],
)

assert resp == [2, 3, 5, 7, 11]
print(resp)
#> [2, 3, 5, 7, 11]
```

## Union

```python
import instructor
import openai
from pydantic import BaseModel
from typing import Union

client = instructor.patch(openai.OpenAI())


class Add(BaseModel):
    a: int
    b: int


class Weather(BaseModel):
    location: str


resp = client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=Union[Add, Weather],
    messages=[
        {
            "role": "user",
            "content": "What is 5 + 5?",
        },
    ],
)

assert resp == Add(a=5, b=5)
print(resp)
#> a=5 b=5
```

## Complex Types

### Pandas DataFrame

```python
from io import StringIO
from typing import Annotated, Any
from pydantic import BeforeValidator, PlainSerializer, InstanceOf, WithJsonSchema
import pandas as pd
import instructor
import openai


def md_to_df(data: Any) -> Any:
    # Convert markdown to DataFrame
    if isinstance(data, str):
        return (
            pd.read_csv(
                StringIO(data),  # Process data
                sep="|",
                index_col=1,
            )
            .dropna(axis=1, how="all")
            .iloc[1:]
            .applymap(lambda x: x.strip())
        )
    return data


MarkdownDataFrame = Annotated[
    InstanceOf[pd.DataFrame],
    BeforeValidator(md_to_df),
    PlainSerializer(lambda df: df.to_markdown()),
    WithJsonSchema(
        {
            "type": "string",
            "description": """
            The markdown representation of the table,
            each one should be tidy, do not try to join
            tables that should be seperate""",
        }
    ),
]


client = instructor.patch(openai.OpenAI())

resp = client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=MarkdownDataFrame,
    messages=[
        {
            "role": "user",
            "content": "Jason is 20, Sarah is 30, and John is 40",
        },
    ],
)

assert isinstance(resp, pd.DataFrame)
print(resp)
"""
        Age
 Name
Jason     20
Sarah     30
John      40
"""
```

### Lists of Unions

```python
import instructor
import openai
from pydantic import BaseModel
from typing import Union, List

client = instructor.patch(openai.OpenAI())


class Weather(BaseModel, frozen=True):
    location: str


class Add(BaseModel, frozen=True):
    a: int
    b: int


resp = client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=List[Union[Add, Weather]],
    messages=[
        {
            "role": "user",
            "content": "Add 5 and 5, and also whats the weather in Toronto?",
        },
    ],
)

assert resp == [Add(a=5, b=5), Weather(location="Toronto")]
print(resp)
#> [Add(a=5, b=5), Weather(location='Toronto')]
```
