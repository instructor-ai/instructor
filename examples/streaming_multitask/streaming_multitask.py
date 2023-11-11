import time

from typing import Iterable
from openai import OpenAI
from pydantic import BaseModel

import instructor


client = instructor.patch(OpenAI())


class User(BaseModel):
    name: str
    job: str
    age: int


def stream_extract(input: str, cls) -> Iterable[User]:
    MultiUser = instructor.MultiTask(cls)
    completion = client.chat.completions.create(
        model="gpt-4-0613",
        temperature=0.1,
        stream=True,
        functions=[MultiUser.openai_schema],
        function_call={"name": MultiUser.openai_schema["name"]},
        messages=[
            {
                "role": "system",
                "content": "You are a perfect entity extraction system",
            },
            {
                "role": "user",
                "content": (
                    f"Consider the data below:\n{input}"
                    "Correctly segment it into entitites"
                    "Make sure the JSON is correct"
                ),
            },
        ],
        max_tokens=1000,
    )
    return MultiUser.from_streaming_response(completion)


start = time.time()
for user in stream_extract(
    input="Create 5 characters from the book Three Body Problem",
    cls=User,
):
    delay = round(time.time() - start, 1)
    print(f"{delay} s: User({user})")
    """
    5.0 s: User(name='Ye Wenjie' job='Astrophysicist' age=50)
    6.6 s: User(name='Wang Miao' job='Nanomaterials Researcher' age=40)
    8.0 s: User(name='Shi Qiang' job='Detective' age=55)
    9.4 s: User(name='Ding Yi' job='Theoretical Physicist' age=45)
    10.6 s: User(name='Chang Weisi' job='Major General' age=60)
    """
    # Notice that the first one would return at 5s bu the last one returned in 10s!
