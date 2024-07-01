---
description: "Self-Ask is a prompting technique that enhances language model performance by encouraging the model to generate and answer follow-up questions before tackling the main query, leading to more accurate and comprehensive responses."
---

By encouraging our model to generate and answer clarifying questions before tackling the main query, we can obtain more accurate and comprehensive responses. This is known as Self-Ask <sup><a href="https://arxiv.org/pdf/2210.03350">1</a></sup>.

We can implement this in Instructor easily as seen below

```python hl_lines="47-49"
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field


client = instructor.from_openai(OpenAI())


class FollowupQuestion(BaseModel):
    question: str = Field(
        ...,
        description="""This is a follow up question
        that the user needs to answer.""",
    )
    answer: str = Field(
        ...,
        description="""This is the
        answer to the above question.""",
    )


class Response(BaseModel):
    follow_up_questions: list[FollowupQuestion] = Field(
        ...,
        description="""These are a list of question and
            answer pairs that are required to be
            answered in order to answer the original
            question.""",
        default_factory=list,
    )
    answer: str = Field(
        ...,
        description="""This is the
            answer to the user's question""",
    )


def generate_response():
    return client.chat.completions.create(
        model="gpt-4o",
        response_model=Response,
        messages=[
            {
                "role": "system",
                "content": """You are a helpful assistant
                which is able to accurately answer user
                queries. Generate a list of follow up
                questions and responses before answering
                the user's question below.""",
            },
            {
                "role": "user",
                "content": """Who was president of the U.S.
                when superconductivity was discovered?""",
            },
        ],
    )


if __name__ == "__main__":
    response = generate_response()
    print(response.follow_up_questions)
    """
    [FollowupQuestion(
        question='When was superconductivity discovered?',
        answer='1911'
    )]
    """
    print(response.answer)
    """
    William Howard Taft was the President of the United States
    when superconductivity was discovered in 1911.
    """
```

### References

<sup id="ref-1">1</sup>: [Measuring and Narrowing the Compositionality Gap in Language Models](https://arxiv.org/pdf/2210.03350)
