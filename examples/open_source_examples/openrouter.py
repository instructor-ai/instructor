import instructor
import openai
from pydantic import BaseModel, Field
from typing import Optional
from instructor import Mode


# Load in your API keys
openrouter_api_key = "OPENROUTER_API_KEY"

# Set your OpenAI base_url to point to Open router base OpenAI API spec endpoint
openrouter_base_url = "https://openrouter.ai/api/v1"

# Create Instructor patched OpenAI client, ensure you're utilizing JSON mode.
client = instructor.patch(openai.OpenAI(api_key=openrouter_api_key, base_url=openrouter_base_url), mode=Mode.JSON)


# Create our simple pydantic UserDetail BaseModel
class UserDetail(BaseModel):
    name: str = Field(..., description="Name of the user")
    age: int
    occupation: Optional[str] = Field(None, description="Occupation of the user")

# Utilizing teknium/openhermes-2.5-mistral-7b, we need to ensure the system prompt is present
model = "teknium/openhermes-2.5-mistral-7b"
system_message = {"role": "system", "content": f"You are an expert at outputting json. Match your response to this json_schema:{UserDetail.model_json_schema()['properties']}"}
user_message = {"role": "user","content": "Extract the user details from the following text: Brandon is 33 years old. He works as a solution architect."}

# Let's call the model
user = client.chat.completions.create(response_format={"type": "json_object"}, model=model, messages=[system_message,user_message])

# Make some assertions
detail = UserDetail.model_validate_json(user.choices[0].message.content)
print(detail)
assert isinstance(detail, UserDetail)
assert detail.name == "Brandon"
assert detail.age == 33