from unittest.mock import MagicMock, patch

import instructor
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice, CompletionUsage
from openai.types.completion_usage import PromptTokensDetails
from pydantic import BaseModel

class TestResponse(BaseModel):
    content: str

def create_mock_completion(cached_tokens: int = 0) -> ChatCompletion:
    prompt_tokens_details = PromptTokensDetails(
        sampled=0,
        cached_tokens=cached_tokens,
        total=20
    )

    usage = CompletionUsage(
        completion_tokens=10,
        prompt_tokens=20,
        total_tokens=30,
        prompt_tokens_details=prompt_tokens_details
    )

    return ChatCompletion(
        id="mock-completion",
        model="gpt-3.5-turbo",
        object="chat.completion",
        created=1234567890,
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    content="Test response",
                    role="assistant"
                ),
                logprobs=None
            )
        ],
        usage=usage
    )

def test_token_details_tracking():
    # Create mock chain
    mock_create = MagicMock()
    mock_completions = MagicMock()
    mock_chat = MagicMock()
    mock_client = MagicMock()

    # Set up the mock chain
    mock_chat.completions = mock_completions
    mock_client.chat = mock_chat
    mock_completions.create = mock_create

    # Create completions for both calls
    completion1 = create_mock_completion(cached_tokens=0)
    completion2 = create_mock_completion(cached_tokens=20)

    # Set up return values for create method
    mock_create.side_effect = [completion1, completion2]

    # Mock both OpenAI and instructor.patch
    with patch('openai.OpenAI', autospec=True) as mock_openai_class, \
         patch('instructor.patch', autospec=True) as mock_patch:

        # Set up the patched client to return our mock
        mock_openai_class.return_value = mock_client

        # Create a mock patched client that returns both response and completion
        mock_patched_client = MagicMock()
        mock_patched_client.chat.completions.create_with_completion.side_effect = [
            (TestResponse(content="Test response"), completion1),
            (TestResponse(content="Test response"), completion2)
        ]

        # Set up the patch to return our mock client
        mock_patch.return_value = mock_patched_client

        # Initialize the client with mock API key
        client = instructor.patch(OpenAI(api_key='mock-api-key'))
        messages = [{"role": "user", "content": "Hello" * 100}]

        # First call should show no cached tokens
        response, completion = client.chat.completions.create_with_completion(
            model="gpt-3.5-turbo",
            messages=messages,
            response_model=TestResponse
        )
        assert completion.usage.prompt_tokens_details.cached_tokens == 0

        # Second call should show cached tokens
        response2, completion2 = client.chat.completions.create_with_completion(
            model="gpt-3.5-turbo",
            messages=messages,
            response_model=TestResponse
        )
        assert completion2.usage.prompt_tokens_details.cached_tokens > 0

        # Verify our mock was called
        assert mock_patched_client.chat.completions.create_with_completion.call_count == 2
