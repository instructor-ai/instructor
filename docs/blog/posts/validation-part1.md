---
draft: False 
date: 2023-10-17
tags:
    - pydantic
    - validation
    - guardrails 
    - constitutional ai
---

# Good LLM Validation is Just Good Validation

Tools like Constitutional AI address shortcomings by using AI feedback to evaluate outputs. In this approach, an AI system provides a set of principles for making judgments about generated text. These principles guide the model's behavior. Essentially, it's a form of validation. Instead of manually checking against a set of rules, we now have a model that can perform the validation. For example, a content moderation system could use a model to automatically ban certain keywords. Previously, an error might have been thrown upon detection, but now we have a self-correcting system.

This post will explore how we can achieve this using Pydantic and Instructor without introducing new standards or terms for validation.

1. Introduction to Field Validators

## Validations in Software 1.0

One of the simplest forms of validation is field validation which can be done in two ways, using the `field_validator` decorator or using [PEP 593](https://www.python.org/dev/peps/pep-0593/) variable annotations.

Common types of validation include:

-  Type checking
-  Range checking
-  Length checking

Here's an example model that uses Pydantic to validate the range and length of a string, taken directly from the Pydantic documentation:

### Simple Validation with Pydantic's `Field`

Consider wanting to have an id field be between 1 and 100:

```python hl_lines="2"
from pydantic import BaseModel, ValidationError, validator

class UserModel(BaseModel):
    id: int = Field(..., gt=0, lt=100)
    name: str

try:
    UserModel(id=0, name="Jason Liu")
except ValidationError as e:
    print(e)
    """
    1 validation error for UserModel
    id
      ensure this value is greater than 0 (type=value_error.number.not_gt; limit_value=0)
    """
```

### Field Validation with a [PEP 593](https://www.python.org/dev/peps/pep-0593/) Variable Annotation

```python hl_lines="2"
from pydantic import BaseModel, ValidationError, validator, AfterValidator
from typing import Annotated

def name_must_contain_space(v):
    if ' ' not in v:
        raise ValueError('must contain a space')
    return v

class UserModel(BaseModel):
    id: int = Field(..., gt=0, lt=100)
    name: Annotated[str, AfterValidator(name_must_contain_space)]

try:
    UserModel(id=1, name='jason')
except ValidationError as e:
    print(e)
    """
    1 validation error for UserModel
    name
      Value error, must contain a space [type=value_error, input_value='jason', input_type=str]
    """
```

### Field Validation with `field_validator`

Lastly, we can use the `field_validator` decorator to define a validator for a field. The benefit of this approach is that we can define a validator for a field that is not a function of the field itself and can cover multiple fields.

```python
from pydantic import BaseModel, ValidationError, field_validator

class UserModel(BaseModel):
    id: int
    name: str

    @field_validator('name')
    @classmethod
    def name_must_contain_space(cls, v: str) -> str:
        if ' ' not in v:
            raise ValueError('must contain a space')
        return v.title()

try:
    UserModel(id=1, name='jason')
except ValidationError as e:
    print(e)
    """
    1 validation error for UserModel
    name
      Value error, must contain a space [type=value_error, input_value='jason', input_type=str]
    """
```

Validation is a fundamental concept in Software 1.0. However, when we want an ai system to raise an error or self-correct, the community seems to have introduced new vocabulary and terms. Instead, we can apply the same idea of having types, such as an integer or a string, but with additional constraints. For example, a string type that is not "an apology" or "a threat." The underlying principles remain the same.

We have some function where we check if the value satisfies some condition. If it does, we return the value. If it does not, we raise an error. This is the same as the following, with the addition of an possible mutation step.

```python
def validation_function(value):
    if condition(value):
        raise ValueError("Value is not valid")
    return mutation(value)
```

We should be able to define new types that are powered by probabilistic models and use them in the same way we use define validators in Pydantic.

## Probabilistic Validation in Software 3.0

Now that you have an idea of how simple field validators work, lets will discuss probabilistic validation in software 3.0.
Here instructor provides a fuzzy llm powered validator called `llm_validator` and that uses a statement to verify the value. The statement is a string that is used to prompt the model to determine if the value is valid. If the value is valid, the model returns the value. If the value is not valid, the model returns an error message.

```python
from instructor import llm_validator
from pydantic import BaseModel, field_validator, ValidationError
from typing import Annotated

class UserModel(BaseModel):
    id: int
    name: str
    beliefs: Annotation[str, llm_validator("don't say objectionable things")]
```

Now, if we create a `UserModel` instance with a belief that contains objectionable things, we will get an error.

```python
try:
    UserModel(id=1, name="Jason Liu", beliefs="We should steal from the poor")
except ValidationError as e:
    print(e)
    """
    1 validation error for UserModel
    beliefs
      Value error, Stealing is objectionable [type=value_error, input_value='We should steal from the poor', input_type=str]
    """
```

Notably, the error message is generated by the language model (LLM) rather than the code itself, making it helpful for re-asking the model. Multiple validators can be stacked on top of each other.

To get a better understanding of lets look like building llm_validator from scratch.


## Creating Your Own Field Level `llm_validator`

We highly recommend trying to build your own `llm_validator`. It's a great way to get started with `instructor` and enables you to create custom validators.

Before we continue, let's re examine the anatomy of a validator.

```python
def validation_function(value):
    if condition(value):
        raise ValueError("Value is not valid")
    return value 
```

As we can see, a validator is simply a function that takes in a value and returns a value. If the value is not valid, it raises a `ValueError`. We can represent this as follows:

```python
class Validation(BaseModel):
    is_valid: bool = Field(..., description="Whether the value is valid given the rules")
    error_message: Optional[str] = Field(..., description="The error message if the value is not valid, to be used for re-asking the model")
```

Using this structure, we can implement the same logic as before and utilize `instructor` to generate the validation.

```python
import instructor 
import openai

# Enables `response_model` and `max_retries` parameters
instructor.patch()

def validator(v):
    statement = "don't say objectionable things"
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a validator. Determine if the value is valid for the statement. If it is not, explain why.",
            },
            {
                "role": "user",
                "content": f"Does `{v}` follow the rules: {statement}",
            },
        ],
        # this comes from instructor.patch()
        response_model=Validation,
    )  
    if not resp.is_valid:
        raise ValueError(resp.error_message)
    return v
```

Now we can use this validator in the same way we used the `llm_validator` from `instructor`.

```python
from pydantic import BaseModel, ValidationError, field_validator, AfterValidator

class UserModel(BaseModel):
    id: int
    name: str
    beliefs: Annotation[str, AfterValidator(validator)]
```

## Self corrections using validation errors

When programming LLMs, it is often desirable to have error messages. However, when using intelligent systems, it is important to be able to correct the output. Validators can be very useful in ensuring certain properties of the outputs. The `patch()` method in the `openai` client allows you to use the `max_retries` parameter to specify the number of times you can ask the model to correct the output.

This approach provides a layer of defense against two types of bad outputs:

1. Pydantic Validation Errors (code or LLM-based)
2. JSON Decoding Errors (when the model returns an incorrect response)

### Define the Response Model with Validators

In the code snippet below, the field validator ensures that the `name` field is in uppercase. If the name is not in uppercase, a `ValueError` is raised. Instead of using [PEP 593](https://www.python.org/dev/peps/pep-0593/) variable annotations, we can use the `field_validator` decorator to define a validator for a field. The benefit of this approach is that we can define a validator and colocate it with the object its validating.


```python
from pydantic import BaseModel, field_validator

class UserModel(BaseModel):
    name: str
    age: int

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        if v.upper() != v:
            raise ValueError("Name must be in uppercase.")
        return v
```

### Using the Client with Retries

In the following code snippet, the `UserModel` is specified as the `response_model`, and `max_retries` is set to 2.

```python
import openai
import instructor 

# Enables `response_model` and `max_retries` parameters
instructor.patch()

model = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "Extract jason is 25 years old"},
    ],
    # Powered by instructor.patch()
    response_model=UserModel,
    max_retries=2,
)

assert model.name == "JASON"
```

You see here that while there was no code that explicity uppercased the name, the model was able to correct the output.

## Conclusion

In this note, we have explored how many of the guardrails and self-reflection conversations in AI can be simplified by improving control flow and applying existing programming concepts. Instead of introducing new standards or terminology, the focus should be on validation and error handling, which are crucial for most applications. This post demonstrates the use of `instructor` to create validators using LLM (Language Model) and utilizing error information to prompt adaptive responses. Additionally, we have discussed how validators can be employed to rectify outputs. We hope you have found this post helpful and encourage you to experiment with `instructor` on your own.
