from typing import Iterable
import openai
import time

from openai_function_call import MultiTask, OpenAISchema


class User(OpenAISchema):
    name: str
    job: str
    age: int


def stream_extract(input: str, cls) -> Iterable[User]:
    MultiUser = MultiTask(cls)
    completion = openai.ChatCompletion.create(
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
    input="Create 10 characters from the book Three Body Problem",
    cls=User,
):
    delay = (time.time() - start) * 100
    print(f"{int(delay)} ms: User({user})")
"""
561 ms: User(name='Ye Wenjie' job='Astrophysicist' age=50)
713 ms: User(name='Wang Miao' job='Nanomaterials Researcher' age=40)
836 ms: User(name='Shi Qiang' job='Detective' age=45)
1001 ms: User(name='Ding Yi' job='Theoretical Physicist' age=42)
1136 ms: User(name='Chang Weisi' job='Major General' age=55)
1274 ms: User(name='Zhang Beihai' job='Space Force Naval Officer' age=52)
1499 ms: User(name='Luo Ji' job='Astronomer' age=48)
1612 ms: User(name='Wei Cheng' job='Mathematician' age=46)
1774 ms: User(name='Shen Yufei' job='Physicist' age=39)
1904 ms: User(name='Pan Han' job='Engineer' age=43)
"""
