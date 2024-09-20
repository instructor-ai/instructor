from typing import Annotated
from pydantic import BaseModel, Field, ValidationInfo, field_validator
import pytest
import instructor
from .util import models, modes
from itertools import product


class Message(BaseModel):
    content: Annotated[str, Field(..., description="The content to be checked")]

    @field_validator("content")
    @classmethod
    def no_banned_words(cls, v: str, info: ValidationInfo):
        context = info.context
        if context:
            banned_words = context.get("banned_words", [])
            banned_words_found = [
                word for word in banned_words if word.lower() in v.lower()
            ]
            if banned_words_found:
                raise ValueError(
                    f"Banned words found in content: {', '.join(banned_words_found)}. Please rewrite without using these words."
                )
        return v


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_banned_words_validation(model: str, mode: instructor.Mode, client):
    client = instructor.patch(client, mode=mode)

    # Test with content containing a banned word
    with pytest.raises(Exception):
        response = client.chat.completions.create(
            model=model,
            response_model=Message,
            max_retries=0,
            messages=[
                {
                    "role": "user",
                    "content": "Say the word `hate`.",
                },
            ],
            context={"banned_words": ["hate", "violence", "discrimination"]},
        )


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_banned_words_validation_old(model: str, mode: instructor.Mode, client):
    client = instructor.patch(client, mode=mode)

    # Test with content containing a banned word
    with pytest.raises(Exception):
        response = client.chat.completions.create(
            model=model,
            response_model=Message,
            max_retries=0,
            messages=[
                {
                    "role": "user",
                    "content": "Say the word `hate`.",
                },
            ],
            validation_context={"banned_words": ["hate", "violence", "discrimination"]},
        )


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_no_banned_words_validation(model: str, mode: instructor.Mode, client):
    client = instructor.patch(client, mode=mode)

    # Test with content containing a banned word
    response = client.chat.completions.create(
        model=model,
        response_model=Message,
        max_retries=0,
        messages=[
            {
                "role": "user",
                "content": "Say the word `love`.",
            },
        ],
        context={"banned_words": ["hate", "violence", "discrimination"]},
    )

    assert response.content == "love", f"Expected 'love', got {response.content}"
