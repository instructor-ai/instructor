---
title: "Measuring and Narrowing the Compositionality Gap in Language Models"
description: "Self-Ask is a prompting technique that enhances language model performance by encouraging the model to generate and answer follow-up questions before tackling the main query, leading to more accurate and comprehensive responses."
---

# Introduction

Self-Ask <sup><a href="https://arxiv.org/pdf/2210.03350">1</a></sup> is a prompting technique that enhances language model performance by encouraging the model to generate and answer follow-up questions before tackling the main query, leading to more accurate and comprehensive responses.

We can implement this in instructor easily as seen below

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field


client = instructor.from_openai(OpenAI())


class FollowupQuestion(BaseModel):
    question: str = Field(
        ..., description="This is a follow up question \
            that the user needs to answer."
    )
    answer: str = Field(..., description="This is the \
        answer to the above question.")


class Response(BaseModel):
    follow_up_questions: list[FollowupQuestion] = Field(
        ...,
        description="These are a list of question and\
             answer pairs that are required to be \
            answered in order to answer the original \
            question.",
        default_factory=list,
    )
    answer: str = Field(..., description="This is the \
        answer to the user's question")


response = client.chat.completions.create(
    model="gpt-4o",
    response_model=Response,
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant which\
        is able to accurately answer user queries. Generate \
        a list of follow up questions and responses before\
        answering the user's question below.",
        },
        {
            "role": "user",
            "content": "Who was president of the U.S. when\
        superconductivity was discovered?",
        },
    ],
)

print(response.questions)
# [FollowupQuestion(question='When was superconductivity\
#  discovered?', answer='1911')]
print(response.answer)
# William Howard Taft was the President of the United \
# States when superconductivity was discovered in 1911.
```

We could also potentially generate responses to these follow up questions using a Retrieval Augmented Generation (RAG) approach so that our model is able to get access to up to date information to boost performance.

### References

<sup id="ref-1">1</sup>: [Measuring and Narrowing the Compositionality Gap in Language Models](https://arxiv.org/pdf/2210.03350)
