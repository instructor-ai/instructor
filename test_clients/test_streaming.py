"""
Test script to verify streaming capabilities across different clients.
This script tests streaming support and documents limitations.
"""

import os
import asyncio
from typing import Optional, AsyncIterator, Dict, Any
from pydantic import BaseModel
import instructor
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion
from anthropic import Anthropic
import google.generativeai as genai
from fireworks.client.openai import OpenAI as FireworksOpenAI
from fireworks.client.openai import AsyncOpenAI as AsyncFireworksOpenAI

class StreamingTestResult(BaseModel):
    """Results of streaming capability tests for a client"""
    client: str
    full_streaming: bool
    partial_streaming: bool
    iterable_streaming: bool
    async_support: bool
    error: Optional[str] = None

class User(BaseModel):
    """Test model for structured output"""
    name: str
    age: int
    bio: Optional[str] = None

async def test_openai_streaming() -> StreamingTestResult:
    """Test OpenAI streaming capabilities"""
    try:
        client = instructor.patch(OpenAI())
        result = StreamingTestResult(
            client="OpenAI",
            full_streaming=False,
            partial_streaming=False,
            iterable_streaming=False,
            async_support=False
        )

        # Test full streaming
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Extract: Jason is 25 years old"}],
                response_model=User,
                stream=True
            )
            async for chunk in response:
                pass
            result.full_streaming = True
        except Exception as e:
            result.error = f"Full streaming failed: {str(e)}"

        # Test partial streaming
        try:
            for partial in client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Extract: Jason is 25 years old"}],
                response_model=User,
                stream=True
            ):
                if isinstance(partial, User):
                    result.partial_streaming = True
                    break
        except Exception as e:
            if not result.error:
                result.error = f"Partial streaming failed: {str(e)}"

        # Test async support
        try:
            async_client = instructor.patch(AsyncOpenAI())
            response = await async_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Extract: Jason is 25 years old"}],
                response_model=User
            )
            if isinstance(response, User):
                result.async_support = True
        except Exception as e:
            if not result.error:
                result.error = f"Async test failed: {str(e)}"

        return result
    except Exception as e:
        return StreamingTestResult(
            client="OpenAI",
            full_streaming=False,
            partial_streaming=False,
            iterable_streaming=False,
            async_support=False,
            error=str(e)
        )

async def test_anthropic_streaming() -> StreamingTestResult:
    """Test Anthropic streaming capabilities"""
    try:
        client = instructor.patch(Anthropic())
        result = StreamingTestResult(
            client="Anthropic",
            full_streaming=False,
            partial_streaming=False,
            iterable_streaming=False,
            async_support=False
        )

        # Test streaming capabilities
        try:
            response = client.messages.create(
                model="claude-3-opus-20240229",
                messages=[{"role": "user", "content": "Extract: Jason is 25 years old"}],
                response_model=User,
                stream=True
            )
            for chunk in response:
                pass
            result.full_streaming = True
        except Exception as e:
            result.error = f"Streaming test failed: {str(e)}"

        return result
    except Exception as e:
        return StreamingTestResult(
            client="Anthropic",
            full_streaming=False,
            partial_streaming=False,
            iterable_streaming=False,
            async_support=False,
            error=str(e)
        )

async def test_fireworks_streaming() -> StreamingTestResult:
    """Test Fireworks streaming capabilities"""
    try:
        client = instructor.patch(FireworksOpenAI())
        result = StreamingTestResult(
            client="Fireworks",
            full_streaming=False,
            partial_streaming=False,
            iterable_streaming=False,
            async_support=False
        )

        # Test streaming
        try:
            response = client.chat.completions.create(
                model="accounts/fireworks/models/llama-v2-7b",
                messages=[{"role": "user", "content": "Extract: Jason is 25 years old"}],
                response_model=User,
                stream=True
            )
            for chunk in response:
                pass
            result.full_streaming = True
        except Exception as e:
            result.error = f"Streaming test failed: {str(e)}"

        # Test async support
        try:
            async_client = instructor.patch(AsyncFireworksOpenAI())
            response = await async_client.chat.completions.create(
                model="accounts/fireworks/models/llama-v2-7b",
                messages=[{"role": "user", "content": "Extract: Jason is 25 years old"}],
                response_model=User
            )
            if isinstance(response, User):
                result.async_support = True
        except Exception as e:
            if not result.error:
                result.error = f"Async test failed: {str(e)}"

        return result
    except Exception as e:
        return StreamingTestResult(
            client="Fireworks",
            full_streaming=False,
            partial_streaming=False,
            iterable_streaming=False,
            async_support=False,
            error=str(e)
        )

async def test_google_streaming() -> StreamingTestResult:
    """Test Google/Gemini streaming capabilities"""
    try:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = instructor.patch(genai.GenerativeModel('gemini-pro'))
        result = StreamingTestResult(
            client="Google/Gemini",
            full_streaming=False,
            partial_streaming=False,
            iterable_streaming=False,
            async_support=False
        )

        # Test streaming
        try:
            response = model.generate_content(
                "Extract: Jason is 25 years old",
                response_model=User,
                stream=True
            )
            for chunk in response:
                pass
            result.full_streaming = True
        except Exception as e:
            result.error = f"Streaming test failed: {str(e)}"

        return result
    except Exception as e:
        return StreamingTestResult(
            client="Google/Gemini",
            full_streaming=False,
            partial_streaming=False,
            iterable_streaming=False,
            async_support=False,
            error=str(e)
        )

async def main() -> None:
    """Run all streaming tests and report results"""
    tests = [
        test_openai_streaming(),
        test_anthropic_streaming(),
        test_fireworks_streaming(),
        test_google_streaming(),
    ]

    results = await asyncio.gather(*tests)

    print("\nStreaming Support Test Results")
    print("=" * 50)
    for result in results:
        print(f"\nClient: {result.client}")
        print(f"Full Streaming: {'✅' if result.full_streaming else '❌'}")
        print(f"Partial Streaming: {'✅' if result.partial_streaming else '❌'}")
        print(f"Iterable Streaming: {'✅' if result.iterable_streaming else '❌'}")
        print(f"Async Support: {'✅' if result.async_support else '❌'}")
        if result.error:
            print(f"Error: {result.error}")
    print("\n")

    # Create a markdown report of the results
    with open("/home/ubuntu/instructor/streaming_support.md", "w") as f:
        f.write("# Streaming Support Status\n\n")
        f.write("| Client | Full Streaming | Partial Streaming | Iterable Streaming | Async Support | Notes |\n")
        f.write("|--------|----------------|------------------|-------------------|---------------|--------|\n")
        for result in results:
            f.write(f"| {result.client} | {'✅' if result.full_streaming else '❌'} | {'✅' if result.partial_streaming else '❌'} | {'✅' if result.iterable_streaming else '❌'} | {'✅' if result.async_support else '❌'} | {result.error or 'No issues'} |\n")

if __name__ == "__main__":
    asyncio.run(main())
