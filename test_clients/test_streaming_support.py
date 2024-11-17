"""Test streaming support for different clients."""
import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Optional, Union, TypeVar
from pydantic import BaseModel
from instructor import Instructor
from instructor.exceptions import InstructorRetryException, IncompleteOutputException

# Type variable for the client
ClientType = TypeVar('ClientType', bound=Instructor)

class StreamingResult(BaseModel):
    """Result of streaming capability test."""
    partial_streaming: bool
    iterable_streaming: bool
    errors: Union[str, None]

class User(BaseModel):
    """Test user model for streaming tests."""
    name: str
    age: int
    bio: str

async def test_streaming_support(
    client: ClientType,
    model_name: str
) -> AsyncIterator[StreamingResult]:
    """Test streaming support for a given client and model.

    Args:
        client: An instructor-patched client instance
        model_name: The name of the model to test

    Yields:
        StreamingResult containing test results and any errors
    """
    try:
        # Test partial streaming
        partial_results: list[User] = []
        try:
            async for partial in client.chat.completions.create_partial(
                model=model_name,
                messages=[
                    {"role": "user", "content": "Create a user profile for Jason, age 25"},
                ],
                response_model=User,
            ):
                if isinstance(partial, User):
                    partial_results.append(partial)
        except (InstructorRetryException, IncompleteOutputException, NotImplementedError) as e:
            yield StreamingResult(
                partial_streaming=False,
                iterable_streaming=False,
                errors=f"Partial streaming not supported: {str(e)}"
            )
            return

        # Test iterable streaming
        iterable_results: list[User] = []
        try:
            users = await client.chat.completions.create_iterable(
                model=model_name,
                messages=[
                    {"role": "user", "content": """
                        Extract users:
                        1. Jason is 25 years old
                        2. Sarah is 30 years old
                    """},
                ],
                response_model=User,
            )

            async for user in users:
                if isinstance(user, User):
                    iterable_results.append(user)
        except (InstructorRetryException, IncompleteOutputException, NotImplementedError) as e:
            yield StreamingResult(
                partial_streaming=len(partial_results) > 0,
                iterable_streaming=False,
                errors=f"Iterable streaming not supported: {str(e)}"
            )
            return

        yield StreamingResult(
            partial_streaming=len(partial_results) > 0,
            iterable_streaming=len(iterable_results) > 0,
            errors=None
        )

    except Exception as e:
        yield StreamingResult(
            partial_streaming=False,
            iterable_streaming=False,
            errors=f"Unexpected error: {str(e)}"
        )

async def test_anthropic_streaming():
    """Test Anthropic's streaming capabilities."""
    try:
        from anthropic import AsyncAnthropic
        import instructor

        client = AsyncAnthropic()
        client = instructor.patch(client)
        async for result in test_streaming_support(client, "claude-3-opus-20240229"):
            return result
    except ImportError:
        return StreamingResult(
            partial_streaming=False,
            iterable_streaming=False,
            errors="Anthropic client not installed"
        )

async def test_openai_streaming():
    """Test OpenAI's streaming capabilities."""
    try:
        from openai import AsyncOpenAI
        import instructor

        client = AsyncOpenAI()
        client = instructor.patch(client)
        async for result in test_streaming_support(client, "gpt-4-turbo-preview"):
            return result
    except ImportError:
        return StreamingResult(
            partial_streaming=False,
            iterable_streaming=False,
            errors="OpenAI client not installed"
        )

async def test_mistral_streaming():
    """Test Mistral's streaming capabilities."""
    try:
        from mistralai.async_client import MistralAsyncClient
        import instructor

        client = MistralAsyncClient()
        client = instructor.patch(client)
        async for result in test_streaming_support(client, "mistral-large-latest"):
            return result
    except ImportError:
        return StreamingResult(
            partial_streaming=False,
            iterable_streaming=False,
            errors="Mistral client not installed"
        )

if __name__ == "__main__":
    # Run tests for each client
    async def main():
        results = {}
        for test_func in [
            test_anthropic_streaming,
            test_openai_streaming,
            test_mistral_streaming
        ]:
            try:
                result = await test_func()
                results[test_func.__name__] = result
            except Exception as e:
                results[test_func.__name__] = StreamingResult(
                    partial_streaming=False,
                    iterable_streaming=False,
                    errors=str(e)
                )
        return results

    results = asyncio.run(main())

    # Generate markdown report
    with open("streaming_support.md", "w") as f:
        f.write("# Client Streaming Support Matrix\n\n")
        f.write("| Client | Partial Streaming | Iterable Streaming | Notes |\n")
        f.write("|--------|------------------|-------------------|--------|\n")

        for test_name, result in results.items():
            client_name = test_name.replace("test_", "").replace("_streaming", "").title()
            partial = "✅" if result.partial_streaming else "❌"
            iterable = "✅" if result.iterable_streaming else "❌"
            notes = result.errors if result.errors else "All features supported"

            f.write(f"| {client_name} | {partial} | {iterable} | {notes} |\n")

        f.write("\n## Notes\n\n")
        f.write("- ✅ = Full support\n")
        f.write("- ❌ = Not supported or failed\n")

    print("\nTest Results:")
    for test_name, result in results.items():
        print(f"\n{test_name}:")
        print(f"Partial Streaming: {result.partial_streaming}")
        print(f"Iterable Streaming: {result.iterable_streaming}")
        if result.errors:
            print(f"Errors: {result.errors}")
