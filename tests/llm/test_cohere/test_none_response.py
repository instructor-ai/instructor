import pytest
from instructor import from_cohere


def test_none_response_model(client):
    client = from_cohere(client, model_name="command-r", max_tokens=1000)

    response = client.messages.create(
        messages=[{"role": "user", "content": "Tell me about your day"}],
        response_model=None,
        temperature=0,
    )

    assert response.text


@pytest.mark.asyncio()
async def test_none_response_model_async(aclient):
    async_client = from_cohere(aclient, model_name="command-r", max_tokens=1000)

    response = await async_client.messages.create(
        messages=[{"role": "user", "content": "Tell me about your day"}],
        response_model=None,
        temperature=0,
    )

    assert response.text
