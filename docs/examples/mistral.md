# Structured Outputs using Mistral
You can now also use mistralai models for inference by using from_mistral.

The examples are using mistral-large-latest.

## MistralAI API
To use mistral you need to obtain a mistral API key.
Goto [mistralai](https://mistral.ai/) click on Build Now and login. Select API Keys from the left menu and then select 
Create API key to create a new key.

## Use example
Some pip packages need to be installed to use the example:
```
pip install instructor mistralai pydantic
```
You need to export the mistral API key:
```
export MISTRAL_API_KEY=<your-api-key>
```

An example:
```python
import os
from pydantic import BaseModel, Field
from typing import List
from mistralai.client import MistralClient
from instructor import from_mistral, Mode

class UserDetails(BaseModel):
    name: str
    age: int


# enables `response_model` in chat call
client = MistralClient(api_key=os.environ.get("MISTRAL_API_KEY"))

instructor_client = from_mistral(client=client, model="mistral-large-latest", 
                                 mode=Mode.MISTRAL_TOOLS, max_tokens=1000)

resp = instructor_client.messages.create(
    response_model=UserDetails,
    messages=[{"role": "user", "content": "Jason is 10"}],
    temperature=0,
)

print(resp)

# output: UserDetails(name='Jason', age=10)
```
