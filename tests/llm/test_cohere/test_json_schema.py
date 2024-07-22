import pytest
from pydantic import BaseModel, Field
import instructor
from instructor.mode import Mode


class User(BaseModel):
    name: str
    age: int


@pytest.mark.parametrize("mode", [Mode.COHERE_JSON_SCHEMA])
def test_parse_user_sync(client, mode):
    client = instructor.from_cohere(client, mode=instructor.Mode.COHERE_JSON_SCHEMA)

    resp = client.chat.completions.create(
        response_model=User,
        messages=[
            {
                "role": "user",
                "content": "Extract user data from this sentence - Ivan is a 27 year old developer from Singapore",
            }
        ],
    )

    assert resp.name == "Ivan"
    assert resp.age == 40


# @pytest.mark.asyncio
# async def test_parse_user_async(aclient):
#     client = instructor.from_cohere(aclient, mode=Mode.COHERE_JSON_SCHEMA)

#     resp = await client.chat.completions.create(
#         response_model=User,
#         messages=[
#             {
#                 "role": "user",
#                 "content": "Extract user data from this sentence - Ivan is a 27 year old developer from Singapore",
#             }
#         ],
#     )

#     assert resp.name == "Ivan"
#     assert resp.age == 27
