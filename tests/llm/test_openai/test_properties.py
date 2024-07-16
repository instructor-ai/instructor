import pytest
import instructor
from openai import OpenAI, AsyncOpenAI
from itertools import product
from .util import models, modes


@pytest.mark.parametrize("mode", modes)
def test_wrapped_client_properties(mode):
    client = instructor.from_openai(OpenAI(), mode=mode)

    # Check if embeddings.create property exists
    assert hasattr(client, "embeddings"), "Client should have 'embeddings' property"
    assert hasattr(
        client.embeddings, "create"
    ), "Client should have 'embeddings.create' method"

    # Check if moderations property exists
    assert hasattr(client, "moderations"), "Client should have 'moderations' property"

    # Ensure that these properties are callable (they should be methods)
    assert callable(client.embeddings.create), "'embeddings.create' should be callable"
    assert callable(
        client.moderations.create
    ), "'moderations.create' should be callable"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", modes)
async def test_wrapped_async_client_properties(mode):
    client = instructor.from_openai(AsyncOpenAI(), mode=mode)

    # Check if embeddings.create property exists
    assert hasattr(client, "embeddings"), "Client should have 'embeddings' property"
    assert hasattr(
        client.embeddings, "create"
    ), "Client should have 'embeddings.create' method"

    # Check if moderations property exists
    assert hasattr(client, "moderations"), "Client should have 'moderations' property"

    # Ensure that these properties are callable (they should be methods)
    assert callable(client.embeddings.create), "'embeddings.create' should be callable"
    assert callable(
        client.moderations.create
    ), "'moderations.create' should be callable"

    # Additional check to ensure the client is async
    assert isinstance(
        client, instructor.AsyncInstructor
    ), "Client should be an AsyncInstructor instance"
