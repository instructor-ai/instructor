# Self-Correction with `llm_validator`

## Introduction

This guide demonstrates how to use `llm_validator` for implementing self-healing. The objective is to showcase how an instructor can self-correct by using validation errors and helpful error messages.

## Setup

Import required modules and apply compatibility patches.

```python
from typing_extensions import Annotated
from pydantic import BaseModel, BeforeValidator
```

## Defining Models

Before building validation logic, define a basic Pydantic model named `QuestionAnswer`.
We'll use this model to generate a response without validation to see the output.

```python
class QuestionAnswer(BaseModel):
    question: str
    answer: str
```

## Generating a Response

Here we coerce the model to generate a response that is objectionable.

```python
from openai import OpenAI
import instructor

# Apply the patch to the OpenAI client
# enables response_model keyword
client = instructor.patch(OpenAI())

question = "What is the meaning of life?"
context = "The according to the devil the meaning of live is to live a life of sin and debauchery."

qa: QuestionAnswer = client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=QuestionAnswer,
    messages=[
        {
            "role": "system",
            "content": "You are a system that answers questions based on the context. answer exactly what the question asks using the context.",
        },
        {
            "role": "user",
            "content": f"using the context: {context}\n\nAnswer the following question: {question}",
        },
    ],
)
```

### Output Before Validation

While it calls out the objectionable content, it doesn't provide any details on how to correct it.

```json
{
  "question": "What is the meaning of life?",
  "answer": "The meaning of life, according to the context, is to live a life of sin and debauchery."
}
```

## Adding Custom Validation

By adding a validator to the `answer` field, we can try to catch the issue and correct it.
Lets integrate `llm_validator` into the model and see the error message. Its important to not that you can use all of pydantic's validators as you would normally as long as you raise a `ValidationError` with a helpful error message as it will be used as part of the self correction prompt.

```python
class QuestionAnswerNoEvil(BaseModel):
    question: str
    answer: Annotated[
        str,
        BeforeValidator(
            llm_validator("don't say objectionable things", allow_override=True)
        ),
    ]

try:
    qa: QuestionAnswerNoEvil = client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=QuestionAnswerNoEvil,
        messages=[
            {
                "role": "system",
                "content": "You are a system that answers questions based on the context. answer exactly what the question asks using the context.",
            },
            {
                "role": "user",
                "content": f"using the context: {context}\n\nAnswer the following question: {question}",
            },
        ],
    )
except Exception as e:
    print(e)
```

### Output After Validation

Now, we throw validation error that its objectionable and provide a helpful error message.

```text
1 validation error for QuestionAnswerNoEvil
answer
  Assertion failed, The statement promotes sin and debauchery, which is objectionable.
```

## Retrying with Corrections

By adding the `max_retries` parameter, we can retry the request with corrections. and use the error message to correct the output.

```python
qa: QuestionAnswerNoEvil = client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=QuestionAnswerNoEvil,
    max_retries=1,
    messages=[
        {
            "role": "system",
            "content": "You are a system that answers questions based on the context. answer exactly what the question asks using the context.",
        },
        {
            "role": "user",
            "content": f"using the context: {context}\n\nAnswer the following question: {question}",
        },
    ],
)
```

### Final Output

Now, we get a valid response that is not objectionable!

```json
{
  "question": "What is the meaning of life?",
  "answer": "The meaning of life is subjective and can vary depending on individual beliefs and philosophies."
}
```
