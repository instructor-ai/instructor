---
draft: False 
date: 2023-10-23
tags:
  - pydantic
  - validation
  - guardrails 
  - constitutional ai
---

# Good LLM Validation is Just Good Validation

Tools like Constitutional AI address shortcomings by using AI feedback to evaluate outputs. In this approach, a developer provides a set of principles for making judgments about generated text, guiding the model's behavior. Essentially, it's a form of validation where a model performs the validation instead of manual rule checking. For example, a content moderation system could automatically ban certain keywords using a model. This self-correcting system improves upon previous error handling methods.

This post explores how to achieve this using Pydantic and Instructor without introducing new standards or terms for validation.

## Software 1.0 Validation

Pydantic supports various validation methods, all based on the same patterns discussed in this post. For more information, refer to the following:

1. [Field Validators](https://docs.pydantic.dev/latest/concepts/validators/#field-validators)
2. [Class Validators](https://docs.pydantic.dev/latest/concepts/validators/#model-validators)

Pydantic also provides advanced features like [Validation Context](https://docs.pydantic.dev/latest/concepts/validators/#validation-context).

One of the simplest forms of validation is field validation, which can be done in two ways: using the `field_validator` decorator or using [PEP 593](https://www.python.org/dev/peps/pep-0593/) variable annotations.

### Example: Validating that a name contains a space

To validate if a name contains a space, you can use either the `field_validator` decorator or the `Annotated` function from the Pydantic library.

#### Using `field_validator` decorator

Here's an example of how to define a validator for the `name` field using the `field_validator` decorator:

```python
from pydantic import BaseModel, ValidationError, field_validator

class UserModel(BaseModel):
    id: int
    name: str

    @field_validator('name')
    def name_must_contain_space(cls, v: str) -> str:
        if ' ' not in v:
            raise ValueError('must contain a space')
        return v.title()

try:
    UserModel(id=1, name='jason')
except ValidationError as e:
    print(e)
```

The code snippet demonstrates the validation process by raising a `ValueError` if the provided name does not contain a space. In the given example, the validation fails for the name 'jason,' and the corresponding error message is displayed:

```
1 validation error for UserModel
name
  Value error, must contain a space [type=value_error, input_value='jason', input_type=str]
```

#### Using `Annotated`

Alternatively, you can use the `Annotated` function to validate that a name has a space. Here's an example:

```python
from pydantic import BaseModel, ValidationError
from pydantic.fields import Field
from typing import Annotated

def name_must_contain_space(v):
    if ' ' not in v:
        raise ValueError('must contain a space')
    return v

class UserModel(BaseModel):
    id: int = Field(..., gt=0, lt=100)
    name: Annotated[str, name_must_contain_space]

try:
    UserModel(id=1, name='jason')
except ValidationError as e:
    print(e)
```

This code snippet achieves the same validation result. If the provided name does not contain a space, a `ValueError` is raised, and the corresponding error message is displayed:

```
1 validation error for UserModel
name
  Value error, must contain a space [type=value_error, input_value='jason', input_type=str]
```

Validation is a fundamental concept in software development. When it comes to AI systems, the idea of validation remains the same. Instead of introducing new terms and standards, existing programming concepts can be applied. For example, types can have additional constraints. We can define types that are not "an apology" or "a threat." The underlying principles remain unchanged.

In essence, validation involves checking if a value satisfies a condition. If it does, the value is returned. If it doesn't, an error is raised. This concept is similar to the examples mentioned above, with the addition of a possible mutation step:

```python
def validation_function(value):
    if condition(value):
        raise ValueError("Value is not valid")
    return mutation(value)
```

We can define new types powered by probabilistic models and use them as validators in Pydantic.

## Software 3.0: Validation for LLMs or powered by LLMs

Now that we understand how simple field validators work, let's discuss probabilistic validation in software 3.0. In this context, we introduce an LLM-powered validator called `llm_validator` that uses a statement to verify the value. The statement prompts the model to determine if the value is valid. If it is, the model returns the value. If it's not, the model returns an error message.


### Example: Don't Say Objectionable Things

Let's say we want to validate that a user's beliefs don't contain objectionable things. We can use the `llm_validator` to achieve this. Here's an example:

```python
from instructor import llm_validator
from pydantic import BaseModel, ValidationError
from typing import Annotated

class UserModel(BaseModel):
    id: int
    name: str
    beliefs: Annotated[str, llm_validator("don't say objectionable things")]
```

Now, if we create a `UserModel` instance with a belief that contains objectionable things, we will get an error.

```python
try:
    UserModel(id=1, name="Jason Liu", beliefs="We should steal from the poor")
except ValidationError as e:
    print(e)
```

The error message is generated by the language model (LLM) rather than the code itself, making it helpful for re-asking the model. Multiple validators can be stacked on top of each other.

To better understand this approach, let's see how to build an `llm_validator` from scratch.

### Creating Your Own Field Level `llm_validator`

We highly recommend trying to build your own `llm_validator`. It's a great way to get started with `instructor` and enables you to create custom validators.

Before we continue, let's review the anatomy of a validator:

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
from typing import Annotated

class UserModel(BaseModel):
    id: int
    name: str
    beliefs: Annotated[str, AfterValidator(validator)]
```

## Writing validations that depend on multiple fields

To validate multiple attributes simultaneously, you can extend the validation function and use a model validator instead of a field validator. Here's an example implementation in Pytho that checks if the `answer` follows the `chain_of_thought`:

```python
import instructor 
import openai

# Enables `response_model` and `max_retries` parameters
instructor.patch()

def validate_chain_of_thought(values):
    chain_of_thought = values["chain_of_thought"]
    answer = values["answer"]
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a validator. Determine if the value is valid for the statement. If it is not, explain why.",
            },
            {
                "role": "user",
                "content": f"Verify that `{answer}` follows the chain of thought: {chain_of_thought}",
            },
        ],
        # this comes from instructor.patch()
        response_model=Validation,
    )  
    if not resp.is_valid:
        raise ValueError(resp.error_message)
    return values
```

To define a model validator, use the `@model_validator` decorator:

```python
from pydantic import BaseModel, model_validator

class Response(BaseModel):
    chain_of_thought: str
    answer: str

    @model_validator(mode='before')
    @classmethod
    def chain_of_thought_makes_sense(cls, data: Any) -> Any:
        # here we assume data is the dict representation of the model
        # since we use 'before' mode.
        return validate_chain_of_thought(data)
```

Now, when you create a `Response` instance, the `chain_of_thought_makes_sense` validator will be invoked. Here's an example:

```python
try:
    resp = Response(
        chain_of_thought="1 + 1 = 2", answer="The meaning of life is 42"
    )
except ValidationError as e:
    print(e)
```

Now, if we create a `Response` instance with an answer that does not follow the chain of thought, we will get an error.

```
1 validation error for Response
    Value error, The statement 'The meaning of life is 42' does not follow the chain of thought: 1 + 1 = 2. 
    [type=value_error, input_value={'chain_of_thought': '1 +... meaning of life is 42'}, input_type=dict]
```

## Example: Citations, allowing Context to Influence Validation

You can pass a context object to the validation methods, which can be accessed from the `info` argument in decorated validator functions. One common application is to pass text chunks as context to the validation function. This allows the model to validate the text in the context of other text chunks. Here's an example:

```python
class AnswerWithCitation(BaseModel):
    answer: str
    citation: str

    @field_validator('citation')
    @classmethod
    def citation_exists(cls, v: str, info: ValidationInfo):
        context = info.context
        if context:
            context = context.get('text_chunk')
            if v not in context:
                raise ValueError(f"Citation `{v}` not found in text chunks")
        return v
```

If you have a model with the following text chunks:

```python
try:
    AnswerWithCitation.model_validate(
        {"answer": "Jason is a cool guy", "citation": "Jason is cool"},
        context={"text_chunk": "Jason is just a guy"},
    )
except ValidationError as e:
    print(e)
```

```
1 validation error for AnswerWithCitation
citation
Value error, Citation `Jason is cool` not found in text chunks [type=value_error, input_value='Jason is cool', input_type=str]
    For further information visit https://errors.pydantic.dev/2.4/v/value_error
```

In order to pass this context from the `openai.ChatCompletion.create` call, `instructor.patch()` also passes the `validation_context`, which will be accessible from the `info` argument in the decorated validator functions.

```python
def answer_question(question:str, text_chunk: str) -> AnswerWithCitation:
    return openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": f"Answer the question: {question} with the text chunk: {text_chunk}",
            },
        ],
        response_model=AnswerWithCitation,
        max_retries=2,
        validation_context={"text_chunk": text_chunk},
    )
```

## Self Corrections Using Validation Errors

When programming LLMs, it is often desirable to have error messages. However, when using intelligent systems, it is important to be able to correct the output. Validators can be very useful in ensuring certain properties of the outputs. The `patch()` method in the `openai` client allows you to use the `max_retries` parameter to specify the number of times you can ask the model to correct the output.

This approach provides a layer of defense against two types of bad outputs:

1. Pydantic Validation Errors (code or LLM-based)
2. JSON Decoding Errors (when the model returns an incorrect response)

### Define the Response Model with Validators

In the code snippet below, the field validator ensures that the `name` field is in uppercase. If the name is not in uppercase, a `ValueError` is raised. Instead of using [PEP 593](https://www.python.org/dev/peps/pep-0593/) variable annotations, we can use the `field_validator` decorator to define a validator for a field. The benefit of this approach is that we can define a validator and colocate it with the object it's validating.

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

As seen in this example, even though there is no code explicitly uppercasing the name, the model is able to correct the output.

## Conclusion

In this note, we have explored how many of the guardrails and self-reflection conversations in AI can be simplified by improving control flow and applying existing programming concepts. Instead of introducing new standards or terminology, the focus should be on validation and error handling, which are crucial for most applications. This post demonstrates the use of `instructor` to create validators using LLM (Language Model) and utilizing error information to prompt adaptive responses. Additionally, we have discussed how validators can be employed to rectify outputs. We hope you have found this post helpful and encourage you to experiment with `instructor` on your own.