from pydantic import BaseModel
from mistralai.client import MistralClient
from instructor import from_mistral
from instructor.function_calls import Mode
import os


class UserDetails(BaseModel):
    name: str
    age: int


# enables `response_model` in chat call
client = MistralClient(api_key=os.environ.get("MISTRAL_API_KEY"))
instructor_client = from_mistral(
    client=client,
    model="mistral-large-latest",
    mode=Mode.MISTRAL_TOOLS,
    max_tokens=1000,
)

resp = instructor_client.messages.create(
    response_model=UserDetails,
    messages=[{"role": "user", "content": "Jason is 10"}],
    temperature=0,
)

print(resp)
