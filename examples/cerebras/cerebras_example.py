import instructor

from cerebras.cloud.sdk import Cerebras
from pydantic import BaseModel
from typing import List

# Define your desired output structure
class UserInfo(BaseModel):
    name: str
    age: int

cerebras_client = Cerebras(api_key=CEREBRAS_API_KEY)
client = instructor.from_cerebras(cerebras_client)

# Extract structured data from natural language
user_info = client.chat.completions.create(
    model="llama3.1-8b",
    response_model=List[UserInfo],
    messages=[{"role": "user", "content": "John Doe is 30 years old."}],
)

for info in user_info:
    print(f"Name: {info.name}, Age: {info.age}")