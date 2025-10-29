"""Tests for streaming with image objects in messages."""

import pytest
from pydantic import BaseModel
import instructor
from instructor.dsl.partial import Partial


class Item(BaseModel):
    name: str
    price: float
    quantity: int


class Receipt(BaseModel):
    items: list[Item]
    total: float


def test_stream_flag_with_non_partial_raises_helpful_error(client):
    """Test that using stream=True with a regular model raises a helpful error."""
    client = instructor.from_openai(client)
    
    with pytest.raises(ValueError) as exc_info:
        client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=Receipt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://templates.mediamodifier.com/645124ff36ed2f5227cbf871/supermarket-receipt-template.jpg"
                            },
                        },
                        {
                            "type": "text",
                            "text": "Analyze the image and return the items in the receipt and the total amount.",
                        },
                    ],
                }
            ],
            stream=True,
        )
    
    error_message = str(exc_info.value)
    assert "Streaming requires using Partial[Model] or Iterable[Model]" in error_message
    assert "create_partial" in error_message
    assert "create_iterable" in error_message


def test_stream_flag_with_non_partial_text_only_raises_helpful_error(client):
    """Test that using stream=True with a regular model raises error even without images."""
    client = instructor.from_openai(client)
    
    with pytest.raises(ValueError) as exc_info:
        client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=Receipt,
            messages=[
                {
                    "role": "user",
                    "content": "Create a fake receipt with 3 items.",
                }
            ],
            stream=True,
        )
    
    error_message = str(exc_info.value)
    assert "Streaming requires using Partial[Model] or Iterable[Model]" in error_message


def test_create_partial_streams_with_image_message(client):
    """Test that create_partial works correctly with image messages."""
    client = instructor.from_openai(client)
    
    result_generator = client.chat.completions.create_partial(
        model="gpt-4o-mini",
        response_model=Receipt,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://templates.mediamodifier.com/645124ff36ed2f5227cbf871/supermarket-receipt-template.jpg"
                        },
                    },
                    {
                        "type": "text",
                        "text": "Analyze the image and return the items in the receipt and the total amount.",
                    },
                ],
            }
        ],
    )
    
    final_result = None
    count = 0
    for partial_result in result_generator:
        assert isinstance(partial_result, Receipt)
        final_result = partial_result
        count += 1
    
    assert count >= 1
    assert final_result is not None
    assert isinstance(final_result, Receipt)
    assert isinstance(final_result.items, list)
    assert isinstance(final_result.total, float)


def test_create_with_partial_response_model_and_stream_works(client):
    """Test that using Partial[Model] directly with stream=True works."""
    client = instructor.from_openai(client)
    
    result_generator = client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=Partial[Receipt],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://templates.mediamodifier.com/645124ff36ed2f5227cbf871/supermarket-receipt-template.jpg"
                        },
                    },
                    {
                        "type": "text",
                        "text": "Analyze the image and return the items in the receipt and the total amount.",
                    },
                ],
            }
        ],
        stream=True,
    )
    
    final_result = None
    count = 0
    for partial_result in result_generator:
        assert isinstance(partial_result, Receipt)
        final_result = partial_result
        count += 1
    
    assert count >= 1
    assert final_result is not None


@pytest.mark.asyncio
async def test_async_stream_flag_with_non_partial_raises_helpful_error(aclient):
    """Test that using stream=True with a regular model raises a helpful error in async."""
    client = instructor.from_openai(aclient)
    
    with pytest.raises(ValueError) as exc_info:
        await client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=Receipt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://templates.mediamodifier.com/645124ff36ed2f5227cbf871/supermarket-receipt-template.jpg"
                            },
                        },
                        {
                            "type": "text",
                            "text": "Analyze the image and return the items in the receipt and the total amount.",
                        },
                    ],
                }
            ],
            stream=True,
        )
    
    error_message = str(exc_info.value)
    assert "Streaming requires using Partial[Model] or Iterable[Model]" in error_message


@pytest.mark.asyncio
async def test_async_create_partial_streams_with_image_message(aclient):
    """Test that async create_partial works correctly with image messages."""
    client = instructor.from_openai(aclient)
    
    result_generator = client.chat.completions.create_partial(
        model="gpt-4o-mini",
        response_model=Receipt,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://templates.mediamodifier.com/645124ff36ed2f5227cbf871/supermarket-receipt-template.jpg"
                        },
                    },
                    {
                        "type": "text",
                        "text": "Analyze the image and return the items in the receipt and the total amount.",
                    },
                ],
            }
        ],
    )
    
    final_result = None
    count = 0
    async for partial_result in result_generator:
        assert isinstance(partial_result, Receipt)
        final_result = partial_result
        count += 1
    
    assert count >= 1
    assert final_result is not None
    assert isinstance(final_result, Receipt)
    assert isinstance(final_result.items, list)
    assert isinstance(final_result.total, float)
