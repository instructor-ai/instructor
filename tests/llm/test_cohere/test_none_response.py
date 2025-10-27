import pytest
import instructor
from instructor import Mode


modes = [
    Mode.COHERE_JSON_SCHEMA,
    Mode.COHERE_TOOLS,
]


@pytest.mark.parametrize("mode", modes)
def test_none_response_model(mode):
    client = instructor.from_provider(
        "cohere/command-a-03-2025",
        max_tokens=1000,
        mode=mode,
    )

    response = client.messages.create(
        messages=[{"role": "user", "content": "Tell me about your day"}],
        response_model=None,
        temperature=0,
    )

    assert response.message


@pytest.mark.asyncio()
@pytest.mark.parametrize("mode", modes)
async def test_none_response_model_async(mode):
    client = instructor.from_provider(
        "cohere/command-a-03-2025",
        max_tokens=1000,
        async_client=True,
        mode=mode,
    )

    response = await client.messages.create(
        messages=[{"role": "user", "content": "Tell me about your day"}],
        response_model=None,
        temperature=0,
    )

    assert response.message
