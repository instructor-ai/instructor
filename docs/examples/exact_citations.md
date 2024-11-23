---
title: Validating AI Answers with Contextual Citations in Python
description: Learn to use Python classes to validate AI-generated answers with citations, ensuring accuracy and preventing hallucinations.
---

# Example: Answering Questions with Validated Citations

For the full code example, check out [examples/citation_fuzzy_match.py](https://github.com/jxnl/instructor/blob/main/examples/citation_with_extraction/citation_fuzzy_match.py)

## Overview

This example shows how to use Instructor with validators to not only add citations to answers generated but also prevent hallucinations by ensuring that every statement made by the LLM is backed up by a direct quote from the context provided, and that those quotes exist!  
Two Python classes, `Fact` and `QuestionAnswer`, are defined to encapsulate the information of individual facts and the entire answer, respectively.

## Data Structures

### The `Fact` Class

The `Fact` class encapsulates a single statement or fact. It contains two fields:

- `fact`: A string representing the body of the fact or statement.
- `substring_quote`: A list of strings. Each string is a direct quote from the context that supports the `fact`.

#### Validation Method: `validate_sources`

This method validates the sources (`substring_quote`) in the context. It utilizes regex to find the span of each substring quote in the given context. If the span is not found, the quote is removed from the list.

```python hl_lines="6 8-13"
from pydantic import Field, BaseModel, field_validator, ConfigDict
from typing import List


class Fact(BaseModel):
    fact: str = Field(...)
    substring_quote: List[str] = Field(...)
    model_config = ConfigDict(validate_default=True)

    @field_validator("substring_quote", mode="after")
    @classmethod
    def validate_sources(cls, substring_quote: List[str], info) -> List[str]:
        text_chunks = info.context.get("text_chunk", None)
        if text_chunks:
            spans = list(cls.get_spans(text_chunks, substring_quote))
            return [text_chunks[span[0] : span[1]] for span in spans]
        return substring_quote

    @staticmethod
    def get_spans(context, quotes):
        for quote in quotes:
            yield from Fact._get_span(quote, context)

    @staticmethod
    def _get_span(quote, context):
        for match in re.finditer(re.escape(quote), context):
            yield match.span()
```

### The `QuestionAnswer` Class

This class encapsulates the question and its corresponding answer. It contains two fields:

- `question`: The question asked.
- `answer`: A list of `Fact` objects that make up the answer.

#### Validation Method: `validate_sources`

This method checks that each `Fact` object in the `answer` list has at least one valid source. If a `Fact` object has no valid sources, it is removed from the `answer` list.

```python hl_lines="5-8"
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List

# <%hide%>
from pydantic import ValidationInfo


class Fact(BaseModel):
    fact: str = Field(...)
    substring_quote: List[str] = Field(...)
    model_config = ConfigDict(validate_default=True)

    @field_validator("substring_quote", mode="after")
    @classmethod
    def validate_sources(cls, substring_quote: List[str], info) -> List[str]:
        text_chunks = info.context.get("text_chunk", None)
        if text_chunks:
            spans = list(cls.get_spans(text_chunks, substring_quote))
            return [text_chunks[span[0] : span[1]] for span in spans]
        return substring_quote

    @staticmethod
    def get_spans(context, quotes):
        for quote in quotes:
            yield from Fact._get_span(quote, context)

    @staticmethod
    def _get_span(quote, context):
        for match in re.finditer(re.escape(quote), context):
            yield match.span()


# <%hide%>
class QuestionAnswer(BaseModel):
    question: str = Field(...)
    answer: List[Fact] = Field(...)
    model_config = ConfigDict(validate_default=True)

    @field_validator("answer", mode="after")
    @classmethod
    def validate_sources(cls, answer: List[Fact]) -> List[Fact]:
        return [fact for fact in answer if len(fact.substring_quote) > 0]
```

## Function to Ask AI a Question

### The `ask_ai` Function

This function takes a string `question` and a string `context` and returns a `QuestionAnswer` object. It uses the OpenAI API to fetch the answer and then validates the sources using the defined classes.

To understand the validation context work from pydantic check out [pydantic's docs](https://docs.pydantic.dev/usage/validators/#model-validators)

```python hl_lines="5 6 14"
from openai import OpenAI
import instructor

# Apply the patch to the OpenAI client
# enables response_model, validation_context keyword
client = instructor.from_openai(OpenAI())


# <%hide%>
from pydantic import ValidationInfo, BaseModel, Field, field_validator, ConfigDict
from typing import List


class Fact(BaseModel):
    fact: str = Field(...)
    substring_quote: List[str] = Field(...)
    model_config = ConfigDict(validate_default=True)

    @field_validator("substring_quote", mode="after")
    @classmethod
    def validate_sources(cls, substring_quote: List[str], info) -> List[str]:
        text_chunks = info.context.get("text_chunk", None)
        if text_chunks:
            spans = list(cls.get_spans(text_chunks, substring_quote))
            return [text_chunks[span[0] : span[1]] for span in spans]
        return substring_quote

    @staticmethod
    def get_spans(context, quotes):
        for quote in quotes:
            yield from Fact._get_span(quote, context)

    @staticmethod
    def _get_span(quote, context):
        for match in re.finditer(re.escape(quote), context):
            yield match.span()


class QuestionAnswer(BaseModel):
    question: str = Field(...)
    answer: List[Fact] = Field(...)
    model_config = ConfigDict(validate_default=True)

    @field_validator("answer", mode="after")
    @classmethod
    def validate_sources(cls, answer: List[Fact]) -> List[Fact]:
        return [fact for fact in answer if len(fact.substring_quote) > 0]


# <%hide%>
def ask_ai(question: str, context: str) -> QuestionAnswer:
    return client.chat.completions.create(
        model="gpt-4-turbo-preview",
        temperature=0,
        response_model=QuestionAnswer,
        messages=[
            {
                "role": "system",
                "content": "You are a world class algorithm to answer questions with correct and exact citations.",
            },
            {"role": "user", "content": f"{context}"},
            {"role": "user", "content": f"Question: {question}"},
        ],
        validation_context={"text_chunk": context},
    )

## Example

dd
Here's an example of using these classes and functions to ask a question and validate the answer.

```python
question = "What did the author do during college?"
context = """
My name is Jason Liu, and I grew up in Toronto Canada but I was born in China.
I went to an arts high school but in university I studied Computational Mathematics and physics.
As part of coop I worked at many companies including Stitchfix, Facebook.
I also started the Data Science club at the University of Waterloo and I was the president of the club for 2 years.
"""
```

The output would be a `QuestionAnswer` object containing validated facts and their sources.

```python
{
    "question": "where did he go to school?",
    "answer": [
        {
            "statement": "Jason Liu went to an arts highschool.",
            "substring_phrase": ["arts highschool"],
        },
        {
            "statement": "Jason Liu studied Computational Mathematics and physics in university.",
            "substring_phrase": ["university"],
        },
    ],
}
```

This ensures that every piece of information in the answer has been validated against the context.
