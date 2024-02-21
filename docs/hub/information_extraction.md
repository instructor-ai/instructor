# Customer Information Extraction

In this guide, we'll walk through how to extract customer lead information using OpenAI's API and Pydantic. This use case is essential for seamlessly automating the process of extracting specific information from a context.

If you want to try this out via `instructor hub`, you can pull it by running:

```bash
instructor hub pull --slug information_extraction --py > information_extraction.py
```

## Motivation

You could potentially integrate this into a chatbot to extract relevant user information from user messages. With the use of machine learning driven validation it would reduce the need for a human to verify the information.

## Defining the Structure

We'll model a customer lead as a Lead object, including fields for the name and phone number. We'll use a Pydantic field_validator for data validation and formatting.

```python
import instructor
import json
from openai import OpenAI
from pydantic import BaseModel, field_validator

class Lead(BaseModel):
    name: str
    phone_number: str

    @field_validator("phone_number")
    @classmethod
    def validate_phone_number(cls, value):
        # Remove any non-digit characters from the phone number
        digits_only = ''.join(filter(str.isdigit, value))

        # Check if the cleaned phone number has exactly 10 digits
        if len(digits_only) != 10:
            raise ValueError("Invalid phone number. Please provide exactly 10 digits.")

        # Format the validated phone number as XXX-XXX-XXXX
        formatted_number = f"{digits_only[:3]}-{digits_only[3:6]}-{digits_only[6:]}"

        return formatted_number

    # Can define some function here to send Lead information to a database using an API
```

## Extracting Lead Information

To extract lead information, we create the `parse_lead_from_message` function which integrates Instructor. It calls OpenAI's API, processes the text, and returns the extracted lead information as a Lead object.

```python
client = instructor.patch(OpenAI())

def parse_lead_from_message(user_message: str):
    return client.chat.completions.create(
        model="gpt-4",
        response_model=Lead,
        messages=[
            {
                "role": "system",
                "content": "You are a data extraction system that extracts a user's name and phone number from a message."
            },
            {
                "role": "user",
                "content": f"Extract the user's lead information from this user's message: {user_message}"
            }
        ]
    )
```

## Evaluating Lead Extraction

To showcase the `parse_lead_from_message` function we can provide sample user messages that may be obtained from a dialogue with a chatbot assistant.

```python
if __name__ == "__main__":
    lead = parse_lead_from_message("Yes, that would be great if someone can reach out my name is Patrick King 9172234587")
    assert isinstance(lead, Lead)
    print(lead)
    print(lead.model_dump(mode="json"))

    # Invalid phone number test
    lead2 = parse_lead_from_message("Yes, that would be great if someone can reach out my name is Patrick King 9172234")
    #assert isinstance(lead2, Lead)
    print(lead2)
    print(lead2.model_dump(mode="json"))

    """
    name='Patrick King' phone_number='917-223-4587'
    {'name': 'Patrick King', 'phone_number': '917-223-4587'}
    
    pydantic_core._pydantic_core.ValidationError: 1 validation error for Lead phone_number Value error, Invalid phone number. Please provide exactly 10 digits. [type=value_error, input_value='9172234', input_type=str] For further information visit https://errors.pydantic.dev/2.6/v/value_error
    """

```

In this example, the `parse_lead_from_message` function successfully extracts lead information from a user message, demonstrating how automation can enhance the efficiency of collecting accurate customer details. It also shows how the function successfully catches that the phone number is invalid so functionality can be implemented for the user to get prompted again to give a correct phone number.