import time

from typing import Iterable
from openai import OpenAI
from pydantic import BaseModel

import instructor


client = instructor.from_openai(OpenAI())


class User(BaseModel):
    name: str
    job: str
    age: int


Users = Iterable[User]


def stream_extract(input: str) -> Users:
    return client.chat.completions.create(
        model="gpt-4-0613",
        temperature=0.1,
        stream=True,
        response_model=Users,
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


start = time.time()
for user in stream_extract(
    input="Create 5 characters from the book Three Body Problem"
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
