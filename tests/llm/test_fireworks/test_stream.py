import pytest
from pydantic import BaseModel
from .util import modes
import instructor
from fireworks.client import Fireworks


@pytest.mark.parametrize("mode, model", modes)
def test_fireworks_streaming(mode: instructor.Mode, model: str):
    class User(BaseModel):
        name: str
        age: int

    client = instructor.from_fireworks(Fireworks(), mode=mode)

    resp = client.chat.completions.create_iterable(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Extract all users from this sentence : Ivan is 27 and lives in Singapore. Darren is around the same age and lives in the same city. Make sure to adhere to the desired JSON format.",
            },
        ],
        response_model=User,
        stream=True,
    )

    users = [user for user in resp]

    assert len(users) == 2
    assert {user.name for user in users} == {"Ivan", "Darren"}
    assert {user.age for user in users} == {27}
