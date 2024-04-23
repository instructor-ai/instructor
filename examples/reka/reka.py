import instructor
from pydantic import BaseModel
import os


client = instructor.from_reka(api_key = os.environ.get("REKA_API_KEY"))
class UserInfo(BaseModel):
    name: str
    age: int


user_info = client.chat.completions.create(
    model="reka-core",
    temperature='0.2',
    response_model=UserInfo,
    messages=[{"role": "user", "content": "Extract John Doe is 30 years old."}],
)

print(user_info.name)
#> John Doe
print(user_info.age)
#> 30