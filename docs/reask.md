# Reasking When Validation Fails

Validators are a great tool for ensuring some property of the outputs. When you use the `patch()` method with the `openai` client, you can use the `max_retries` parameter to set the number of times you can reask. This allows the client to reattempt the API call a specified number of times if validation fails. Its another layer of defense against bad outputs of two forms.

1. Pydantic Validation Errors
2. JSON Decoding Errors

## Future Improvements

!!! notes "Contributions Welcome"
    The current retry mechanism relies on a while loop. For a more robust solution, contributions to integrate the `tenacity` library are welcome.

## Example: Using Validators for Reasking

The example utilizes Pydantic's field validators in tandem with the `max_retries` parameter. In this example if the `name` field fails validation, the `openai` client will reattempt the API call. Here we use a plain validator, but we can also use [llms for validation](validation.md)

### Step 1: Define the Response Model with Validators

```python
import instructor
from pydantic import BaseModel, field_validator

# Apply the patch to the OpenAI client
instructor.patch()

class UserDetails(BaseModel):
    name: str
    age: int

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        if v.upper() != v:
            raise ValueError("Name must be in uppercase.")
        return v
```

Here, the `UserDetails` class includes a validator for the `name` attribute. The validator checks that the name is in uppercase and raises a `ValueError` otherwise.

### Step 2: Exception Handling and Reasking

When validation fails, several steps are taken:

1. The existing messages are retained for the new API request.
2. The previous function call's response is added back.
3. A user prompt is included to reask the model, with details on the error.

```python
try:
    ...
except (ValidationError, JSONDecodeError) as e:
    kwargs["messages"].append(dict(**response.choices[0].message))
    kwargs["messages"].append(
        {
            "role": "user",
            "content": f"Please correct the function call; errors encountered:\n{e}",
        }
    )
```

## Using the Client with Retries

Here, the `UserDetails` model is passed as the `response_model`, and `max_retries` is set to 2.

```python
model = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    response_model=UserDetails,
    max_retries=2,
    messages=[
        {"role": "user", "content": "Extract jason is 25 years old"},
    ],
)

assert model.name == "JASON"
```

The `max_retries` parameter will trigger up to 2 reattempts if the `name` attribute fails the uppercase validation in `UserDetails`.

## Takeaways

Instead of framing "self-critique" or "self-reflection" in AI as new concepts, we can view them as validation errors with clear error messages that the systen can use to heal. This approach leverages existing programming practices for error handling, avoiding the need for new methodologies. We simplify the issue into code we already know how to write and leverage pydantic's powerful validation system to do so.