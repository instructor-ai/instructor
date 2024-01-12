import json
import instructor
import asyncio

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, field_validator
from typing import List
from enum import Enum
import diskcache
import os
import inspect
import functools

client = instructor.patch(AsyncOpenAI(), mode=instructor.Mode.TOOLS)
sem = asyncio.Semaphore(5)

pwd = os.getcwd()
cache = diskcache.Cache(pwd)


def instructor_cache(func):
    """Cache a function that returns a Pydantic model"""
    return_type = inspect.signature(func).return_annotation  #
    if not issubclass(return_type, BaseModel):  #
        raise ValueError("The return type must be a Pydantic model")

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        key = f"{func.__name__}-{functools._make_key(args, kwargs, typed=False)}"
        if (cached := cache.get(key)) is not None:
            # Deserialize from JSON based on the return type
            return return_type.model_validate_json(cached)

        result = await func(*args, **kwargs)
        # Call the function and cache its result

        serialized_result = result.model_dump_json()
        cache.set(key, serialized_result)

        return result

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        key = f"{func.__name__}-{functools._make_key(args, kwargs, typed=False)}"
        if (cached := cache.get(key)) is not None:
            return return_type.model_validate_json(cached)

        result = func(*args, **kwargs)
        serialized_result = result.model_dump_json()
        cache.set(key, serialized_result)

        return result

    return wrapper if inspect.iscoroutinefunction(func) else sync_wrapper


class QuestionType(Enum):
    CONTENT_OWNERSHIP = "CONTENT_OWNERSHIP"
    CONTACT = "CONTACT"
    TIMELINE_QUERY = "TIMELINE_QUERY"
    DOCUMENT_SEARCH = "DOCUMENT_SEARCH"
    COMPARE_CONTRAST = "COMPARE_CONTRAST"
    MEETING_TRANSCRIPTS = "MEETING_TRANSCRIPTS"
    EMAIL = "EMAIL"
    PHOTOS = "PHOTOS"
    HOW_DOES_THIS_WORK = "HOW_DOES_THIS_WORK"
    NEEDLE_IN_HAYSTACK = "NEEDLE_IN_HAYSTACK"
    SUMMARY = "SUMMARY"


ALLOWED_TYPES = [t.value for t in QuestionType]


# You can add more instructions and examples in the description
# or you can put it in the prompt in `messages=[...]`
class QuestionClassification(BaseModel):
    """
    Predict the type of question that is being asked.

    Here are some tips on how to predict the question type:

    CONTENT_OWNERSHIP: "Who owns the a certain piece of content?"
    CONTACT: Searches for some contact information.
    TIMELINE_QUERY: "When did something happen?
    DOCUMENT_SEARCH: "Find me a document"
    COMPARE_CONTRAST: "Compare and contrast two things"
    MEETING_TRANSCRIPTS: "Find me a transcript of a meeting, or a soemthing said in a meeting"
    EMAIL: "Find me an email, search for an email"
    PHOTOS: "Find me a photo, search for a photo"
    HOW_DOES_THIS_WORK: "How does this question /answer product work?"
    NEEDLE_IN_HAYSTACK: "Find me something specific in a large amount of data"
    SUMMARY: "Summarize a large amount of data"
    """

    # If you want only one classification, just change it to
    #   `classification: QuestionType` rather than `classifications: List[QuestionType]``
    classification: List[QuestionType] = Field(
        description=f"An accuracy and correct prediction predicted class of question. Only allowed types: {ALLOWED_TYPES}, should be used",
    )

    @field_validator("classification", mode="before")
    def validate_classification(cls, v):
        # sometimes the API returns a single value, just make sure it's a list
        if not isinstance(v, list):
            v = [v]
        return v


@instructor_cache
async def classify_question(user_question: str) -> QuestionClassification:
    return await client.chat.completions.create(
        model="gpt-4",
        response_model=QuestionClassification,
        max_retries=2,
        messages=[
            {
                "role": "user",
                "content": f"Classify the following question: {user_question}",
            },
        ],
    )


async def classify(data: str) -> QuestionClassification:
    async with sem:  # some simple rate limiting
        return data, await classify_question(data)


async def main(
    questions: List[str], *, path_to_jsonl: str = None
) -> List[QuestionClassification]:
    tasks = [classify(question) for question in questions]
    for task in asyncio.as_completed(tasks):
        question, label = await task
        resp = {
            "question": question,
            "classification": [c.value for c in label.classification],
        }
        print(resp)
        if path_to_jsonl:
            with open(path_to_jsonl, "a") as f:
                json_dump = json.dumps(resp)
                f.write(json_dump + "\n")


if __name__ == "__main__":
    import asyncio

    path = "./data.jsonl"

    questions = [
        "What was that ai app that i saw on the news the other day?",
        "What was that ai app that i saw on the news the other day?",
        "What was that ai app that i saw on the news the other day?",
    ]

    asyncio.run(main(questions, path_to_jsonl=path))
