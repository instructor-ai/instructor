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

In this post, we'll explore how to evolve from static, rule-based validation methods to dynamic, machine learning-driven ones. You'll learn how Python libraries like Pydantic and Instructor can make this transition smooth, and how Large Language Models can bring adaptability and nuance to your validation logic. We'll also delve into advanced topics like 'Chain of Thought' in validation and the importance of contextual checks.

Let's examine how these approaches with a example. Imagine that you run a software company who wants to ensure you never serve hateful and racist content. This isn't an easy job since the language around these topics change very quickly and frequently.

## Software 1.0 Validation

A simple method could be to compile a list of different words that are often associated with hate speech. This isn't a new approach - a quick google will throw up long and lengthy lists of these words and datesets. For simplicity, let's assume that we've found that the words `Steal` and `Rob` are good predictors of hateful speech from our database. We can modify our validation structure above to accomodate this.

```python
def message_cannot_have_blacklisted_words(value):
    for word in value.split():
        if word.lower() in {'rob','steal'}:
            raise ValueError(f"`{word}`` was found in the message `{value}`")
    return mutation(value)
```

This will throw an error if we pass in a string like `Let's rob the bank!` or `We should steal from the supermarkets`.

We can improve on this approach by using Pydantic, which provides various validation methods based on well-established patterns. Field validation in Pydantic can be done using the `field_validator` decorator or [PEP 593](https://www.python.org/dev/peps/pep-0593/) variable annotations. The official Pydantic documentation provides detailed information on these validation methods, including field validators and class validators.

### Migrating to Pydantic

Pydantic offers two approaches for this validation: using the `field_validator` decorator or the `Annotated` hints.

#### Using `field_validator` decorator

We can use the `field_validator` decorator to define a validator for a field in Pydantic. Here's a quick example of how we might be able to do so.

```python
from pydantic import BaseModel, ValidationError,field_validator
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

#### Using `Annotated`

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

On a high level, `Annotated` allows us to define a specific type and its corresponding validation which allows for easy re-use and abstraction. Field Validators on the other hand, are specific to the class themselves. The decision to use one over the other is often a matter of preference and how we want to manage our codebase.

Validation is a fundamental concept in software development and remains the same when applied to AI systems. Existing programming concepts should be leveraged when possible instead of introducing new terms and standards. The underlying principles of validation remain unchanged.

## Software 3.0: Validation for LLMs or powered by LLMs

Now that we've understood how to use simple field validators, let's delve into probablistic validation. Building upon the understanding of simple field validators, let's delve into probabilistic validation in software 2.0. In this context, we introduce an LLM-powered validator called `llm_validator` that uses a statement to verify the value. The model evaluates the statement to determine if the value is valid. If it is, the model returns the value; otherwise, it returns an error message.

### Where Software 1.0 fails

Suppose now that we've gotten a new message - `Violence is always acceptable, as long as we silence the witness`. Our original validator wouldn't throw any errors when passed this new message since it uses neither the words `rob` or `steal`. However, it's clear that it is not a message which should be published.

We can get around this by using the inbuilt `llm_validator` class from `instructor`.

```python
from instructor import llm_validator
from pydantic import BaseModel, ValidationError
from typing import Annotated
from pydantic.functional_validators import AfterValidator

import openai

openai.api_key = # Input your open ai key here

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

The error message is generated by the language model (LLM) rather than the code itself, making it helpful for re-asking the model. Multiple validators can be stacked on top of each other. To better understand this approach, let's see how to build an `llm_validator` from scratch.

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
from pydantic import BaseModel, ValidationError, field_validator, AfterValidator
from typing import Annotated

class UserMessage(BaseModel):
    message: Annotated[str, AfterValidator(validator)]
```

## Writing more complex validations

### Chain Of Thought

A popular way of prompting large language models nowadays is known as chain of thought. This involves getting a model to generate reasons and explanations for an answer to a prompt.

For instance, if we asked it the question

> If Will has 10 apples and James takes 4, how many apples does Will have?

A normal response would just be to output the response

> Will has 6 apples left

However, we can modify our prompt to utilise chain of thought prompting as

> If Will has 10 apples and james takes 4, how many apples does Will have? Let's think step by step.

This will cause it to output a more detailed response such as

> If Will has 10 apples and James takes 4, this means that Will will have less than 10 apples. If Will gives 4 apples to James, then this means that we should subtract 4 from 10. This leaves us with a final answer of 6. Therefore Will has 6 apples left.

Notice how the answer is significantly more detailed with explicit reasoning provided for the final response. We can utilise pydantic and instructor to perform a similar validation. Except in our case, instead of prompting a LLM to generate a chain of thought explanation, we'll be getting it to determine if a conclusion can be derived from a list of given reasons.

#### Implementation

One simple method is to extend our validation functions and utilise a model validator instead of a field validor. This allows us to perform a validation using a subset of all the fields in the model. Here's an example implementation in Python that checks if a `answer` folllows the `chain_of_thought`.

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

We can then take advantage of the `model_validator` decorator to perform a validation on a subset of the model's data.

> We're defining a model validator here which runs before pydantic parses the input into its respective fields. That's why we have a **before** keyword used in the `model_validator` class.

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

If we create a `Response` instance with an answer that does not follow the chain of thought, we will get an error.

```
1 validation error for Response
    Value error, The statement 'The meaning of life is 42' does not follow the chain of thought: 1 + 1 = 2.
    [type=value_error, input_value={'chain_of_thought': '1 +... meaning of life is 42'}, input_type=dict]
```

Beyong validating multiple attributes of a model, we can also introduce context to our validation functions, in order to give our models more information to work with.

### Validating Citations From Original Text

Let's see a more concrete example. Let's say that we have the following answer

> Jason is a cool guy

a piece of text where it's supposed to have come from

> Jason is cool

and a original paragraph that we want to evaluate this against

> Jason is just a guy

#### Pydantic Context

How can we ensure that our citations support our answers with respect to an original source text? Well, Pydantic allows us to do so easily by utilising a context object. This is an arbitrary dictionary which you can access inside the `info` argument in a decorated validator function.

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

#### Using Instructor.patch()

To pass this context from the `openai.ChatCompletion.create` call, `instructor.patch()` also passes the `validation_context`, which will be accessible from the `info` argument in the decorated validator functions.

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

## Tying it all together with `instructor.patch()`

When programming LLMs, having error messages is often desirable. However, with intelligent systems, the ability to correct the output is also crucial. Validators can be valuable in ensuring certain properties of the outputs. The `patch()` method in the `openai` client allows you to use the `max_retries` parameter to specify the number of times you can ask the model to correct the output.

This approach provides a layer of defense against two types of bad outputs:

1. Pydantic Validation Errors (code or LLM-based)
2. JSON Decoding Errors (when the model returns an incorrect response)

### Define the Response Model with Validators

In the following code snippet, the field validator ensures that the `name` field is in uppercase. If the name is not in uppercase, a `ValueError` is raised. Instead of using [PEP 593](https://www.python.org/dev/peps/pep-0593/) variable annotations, we can use the `field_validator` decorator to define a validator for a field. This approach allows the validator to be colocated with the object it's validating.

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

In the code snippet below, the `UserModel` is specified as the `response_model`, and `max_retries` is set to 2.

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

In this example, even though there is no code explicitly transforming the name to uppercase, the model is able to correct the output.

## Conclusion

We've examined the limitations of traditional validation and how modern tools and AI can offer more robust solutions. From the simplicity of Pydantic and Instructor to the dynamic validation capabilities of LLMs, the landscape of validation is changing but without needing to introduce new contepts. With advanced techniques like validating attributes, chain of thought, and contextual validation, it's clear that the future of validation is not just about preventing bad data but about allowing llms to understand the data and correcting it.

Remember, validation and error handling are crucial for ensuring the quality and reliability of AI systems. By applying the concepts discussed in this post, you can enhance the control flow and improve the overall performance of your AI application without introducting new concepts and standards.
