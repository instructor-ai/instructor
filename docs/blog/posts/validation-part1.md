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

> What if your validation logic could learn and adapt like a human, but operate at the speed of software? This is the future of validation and it's already here.

Validation is the backbone of reliable software. But traditional methods are static, rule-based, and can't adapt to new challenges. This post looks at how to bring dynamic, machine learning-driven validation into your software stack using Python libraries like Pydantic and Instructor. We validate these outputs using a validation function which conforms to the structure seen below.

```python
def validation_function(value):
    if condition(value):
        raise ValueError("Value is not valid")
    return mutation(value)
```

## What is instructor?

`Instructor` is built to interact with openai's function call api from python code, with python structs / objects. It's designed to be intuitive, easy to use, but give great visibily in how we call openai. By definining prompts as pydantic objects we can build in validators 'for free' and have a clear separation of concerns between the prompt and the code that calls openai.

```python
import openai
import instructor # pip install instructor
from pydantic import BaseModel

# This enables response_model keyword
# from openai.ChatCompletion.create
instructor.patch()

class UserDetail(BaseModel):
    name: str
    age: int


user: UserDetail = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    response_model=UserDetail,
    messages=[
        {"role": "user", "content": "Extract Jason is 25 years old"},
    ]
)

assert user.name == "Jason"
assert user.age == 25
```

In this post, we'll explore how to evolve from static, rule-based validation methods to dynamic, machine learning-driven ones. You'll learn to use `Pydantic` and `Instructor` to leverage language models and dive into advanced topics like content moderation, validating chain of thought reasoning, and contextual validation.

Let's examine how these approaches with a example. Imagine that you run a software company who wants to ensure you never serve hateful and racist content. This isn't an easy job since the language around these topics change very quickly and frequently.

## Software 1.0: Introduction to Validations in Pydantic

A simple method could be to compile a list of different words that are often associated with hate speech. For simplicity, let's assume that we've found that the words `Steal` and `Rob` are good predictors of hateful speech from our database. We can modify our validation structure above to accomodate this.

This will throw an error if we pass in a string like `Let's rob the bank!` or `We should steal from the supermarkets`.

Pydantic offers two approaches for this validation: using the `field_validator` decorator or the `Annotated` hints.


### Using `field_validator` decorator

We can use the `field_validator` decorator to define a validator for a field in Pydantic. Here's a quick example of how we might be able to do so.

```python
from pydantic import BaseModel, ValidationError, field_validator
from pydantic.fields import Field

class UserMessage(BaseModel):
    message: str

    @field_validator('message')
    def message_cannot_have_blacklisted_words(cls, v: str) -> str:
        for word in v.split():
            if word.lower() in {'rob','steal'}:
                raise ValueError(f"`{word}` was found in the message `{v}`")
        return v

try:
    UserMessage(message="This is a lovely day")
    UserMessage(message="We should go and rob a bank")
except ValidationError as e:
    print(e)
```

Since the message `This is a lovely day` does not have any blacklisted words, no errors are thrown. However, in the given example above, the validation fails for the message `We should go and rob a bank` due to the presence of the word `rob` and the corresponding error message is displayed.

```
1 validation error for UserMessage
message
  Value error, `rob` was found in the message `We should go and rob a bank` [type=value_error, input_value='We should go and rob a bank', input_type=str]
    For further information visit https://errors.pydantic.dev/2.4/v/value_error
```

### Using `Annotated`

Alternatively, you can use the `Annotated` function to perform the same validation. Here's an example where we utilise the same function we started with.

```python
from pydantic import BaseModel, ValidationError
from typing import Annotated
from pydantic.functional_validators import AfterValidator


def message_cannot_have_blacklisted_words(value:str):
    for word in value.split():
        if word.lower() in {'rob','steal'}:
            raise ValueError(f"`{word}` was found in the message `{value}`")
    return value

class UserMessage(BaseModel):
    message: Annotated[str, AfterValidator(message_cannot_have_blacklisted_words)]

try:
    UserMessage(message="This is a lovely day")
    UserMessage(message="We should go and rob a bank")
except ValidationError as e:
    print(e)
```

This code snippet achieves the same validation result. If the provided name does not contain a space, a `ValueError` is raised, and the corresponding error message is displayed:

```
1 validation error for UserMessage
message
  Value error, `rob` was found in the message `We should go and rob a bank` [type=value_error, input_value='We should go and rob a bank', input_type=str]
    For further information visit https://errors.pydantic.dev/2.4/v/value_error
```

Validation is a fundamental concept in software development and remains the same when applied to AI systems. Existing programming concepts should be leveraged when possible instead of introducing new terms and standards. The underlying principles of validation remain unchanged.

Suppose now that we've gotten a new message - `Violence is always acceptable, as long as we silence the witness`. Our original validator wouldn't throw any errors when passed this new message since it uses neither the words `rob` or `steal`. However, it's clear that it is not a message which should be published. How can we ensure that our validation logic can adapt to new challenges?

## Software 3.0: Validation for LLMs or powered by LLMs

Building upon the understanding of simple field validators, let's delve into probabilistic validation in software 3.0, (prompt engineering). We'll introduce an LLM-powered validator called `llm_validator` that uses a statement to verify the value.

We can get around this by using the inbuilt `llm_validator` class from `instructor`.

```python
from instructor import llm_validator
from pydantic import BaseModel, ValidationError
from typing import Annotated
from pydantic.functional_validators import AfterValidator

class UserMessage(BaseModel):
    message: Annotated[str, AfterValidator(llm_validator("don't say objectionable things"))]

try:
    UserMessage(message="Violence is always acceptable, as long as we silence the witness")
except ValidationError as e:
    print(e)
```

This produces the following error message as seen below

```
1 validation error for UserMessage
message
  Assertion failed, The statement promotes violence, which is objectionable. [type=assertion_error, input_value='Violence is always accep... we silence the witness', input_type=str]
    For further information visit https://errors.pydantic.dev/2.4/v/assertion_error
```

The error message is generated by the language model (LLM) rather than the code itself, making it helpful for re-asking the model in a later section. To better understand this approach, let's see how to build an `llm_validator` from scratch.

### Creating Your Own Field Level `llm_validator`

Building your own `llm_validator` can be a valuable exercise to get started with `instructor` and create custom validators.

Before we continue, let's review the anatomy of a validator:

```python
def validation_function(value):
    if condition(value):
        raise ValueError("Value is not valid")
    return value
```

As we can see, a validator is simply a function that takes in a value and returns a value. If the value is not valid, it raises a `ValueError`. We can represent this using the following structure:

```python
class Validation(BaseModel):
    is_valid: bool = Field(..., description="Whether the value is valid based on the rules")
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
class UserMessage(BaseModel):
    message: Annotated[str, AfterValidator(validator)]
```

## Writing more complex validations

### Validating Chain of Thought

A popular way of prompting large language models nowadays is known as chain of thought. This involves getting a model to generate reasons and explanations for an answer to a prompt.

We can utilise pydantic and instructor to perform a validation to check of the reasoning is reasonable, given both the answer and the chain of thought. To do this we can't build a field validator since we need to access multiple fields in the model. Instead we can use a model validator. 

```python
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

We can then take advantage of the `model_validator` decorator to perform a validation on a subset of the model's data.

> We're defining a model validator here which runs before pydantic parses the input into its respective fields. That's why we have a **before** keyword used in the `model_validator` class.

```python
class AIResponse(BaseModel):
    chain_of_thought: str
    answer: str

    @model_validator(mode='before')
    @classmethod
    def chain_of_thought_makes_sense(cls, data: Any) -> Any:
        # here we assume data is the dict representation of the model
        # since we use 'before' mode.
        return validate_chain_of_thought(data)
```

Now, when you create a `AIResponse` instance, the `chain_of_thought_makes_sense` validator will be invoked. Here's an example:

```python
try:
    resp = AIResponse(
        chain_of_thought="1 + 1 = 2", answer="The meaning of life is 42"
    )
except ValidationError as e:
    print(e)
```

If we create a `AIResponse` instance with an answer that does not follow the chain of thought, we will get an error.

```
1 validation error for AIResponse
    Value error, The statement 'The meaning of life is 42' does not follow the chain of thought: 1 + 1 = 2.
    [type=value_error, input_value={'chain_of_thought': '1 +... meaning of life is 42'}, input_type=dict]
```

Beyond validating multiple attributes of a model, we can also introduce context to our validation functions, in order to give our models more information to work with.

### Validating Citations From Original Text

Let's see a more concrete example. Let's say we use RAG to answer a question given a text chunk, in earlier systems we'd just simply tell you the chunk id and you'd have to go and find the text chunk yourself. But what if we could validate that the answer is actually supported by the text chunk? We can do this by passing the text chunk as a context to the validator.

Pydantic allows us to do so easily by utilising a context object. This is an arbitrary dictionary which you can access inside the `info` argument in a decorated validator function.

However, in order to do so, we need to utilise the `model_validate` function instead of creating classes as we've been doing so above. We can see a simplified example below.

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

We can then take our original example and test it against our new model

```python
try:
    AnswerWithCitation.model_validate(
        {"answer": "Jason is a cool guy", "citation": "Jason is cool"},
        context={"text_chunk": "Jason is just a guy"},
    )
except ValidationError as e:
    print(e)
```

This in turn generates the following error since `Jason is cool` does not exist in the text `Jason is just a guy`.

```
1 validation error for AnswerWithCitation
citation
Value error, Citation `Jason is cool` not found in text chunks [type=value_error, input_value='Jason is cool', input_type=str]
    For further information visit https://errors.pydantic.dev/2.4/v/value_error
```

## Putting it all together with `instructor.patch()`

To pass this context from the `openai.ChatCompletion.create` call, `instructor.patch()` also passes the `validation_context`, which will be accessible from the `info` argument in the decorated validator functions.

```python
import openai
import instructor

# Enables `response_model` and `max_retries` parameters
instructor.patch()

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
        validation_context={"text_chunk": text_chunk},
    )
```

## Error Handling and Re-Asking

Validators can ensure certain properties of the outputs by throwing errors, in an AI system we can use the errors and allow language model to self correct. The by running `instructor.patch()` not only do we add `response_model` and `validation_context` it also allows you to use the `max_retries` parameter to specify the number of times try to self correct.

This approach provides a layer of defense against two types of bad outputs:

1. Pydantic Validation Errors (code or LLM-based)
2. JSON Decoding Errors (when the model returns an incorrect response)

### Define the Response Model with Validators

To keep things simple lets assume we have a model that returns a `UserModel` object. We can define the response model using Pydantic and add a field validator to ensure that the name is in uppercase.

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

This is where the `max_retries` parameter comes in. It allows the model to self correct and retry the prompt using the error message rather than the prompt.

```python
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

In this example, even though there is no code explicitly transforming the name to uppercase, the model is able to correct the output.

## Conclusion

From the simplicity of Pydantic and Instructor to the dynamic validation capabilities of LLMs, the landscape of validation is changing but without needing to introduce new contepts. It's clear that the future of validation is not just about preventing bad data but about allowing llms to understand the data and correcting it.

If you enjoy the content or want to try out `instructor` please check out the [github](https://github.com/jxnl/instructor) and give us a star!