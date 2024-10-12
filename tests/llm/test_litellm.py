import instructor
from litellm import acompletion, completion


def test_litellm_create():
    client = instructor.from_litellm(completion)

    assert(isinstance(client, instructor.Instructor))

def test_async_litellm_create():
    client = instructor.from_litellm(acompletion)

    assert(isinstance(client, instructor.AsyncInstructor))
