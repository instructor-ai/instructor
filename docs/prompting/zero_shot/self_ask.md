---
description: "Self-Ask is a prompting technique that enhances language model performance by encouraging the model to generate and answer follow-up questions before tackling the main query, leading to more accurate and comprehensive responses."
---

By encouraging our model to generate and answer clarifying questions before tackling the main query, we can obtain more accurate and comprehensive responses. This is known as Self-Ask <sup><a href="http://users.umiacs.umd.edu/~jbg/docs/2023_findings_more.pdf">1</a></sup>.

We can implement this using `instructor` as seen below.

```python hl_lines="35-37"
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field


client = instructor.from_openai(OpenAI())


class FollowupQuestion(BaseModel):
    question: str = Field(
        description="Question to be answered",
    )
    answer: str

class SelfAskResponse(BaseModel):
    follow_up_questions: list[FollowupQuestion] = Field(
        description="""A list of question and
            answer pairs that are required to be
            answered in order to answer the original
            question.""",
        default_factory=list,
    )
    answer:str


def generate_questions_and_response(query):
    return client.chat.completions.create(
        model="gpt-4o",
        response_model=SelfAskResponse,
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
                "content": query,
            },
        ],
    )


if __name__ == "__main__":
    query = """Who was president of the U.S.
    when superconductivity was discovered?"""

    response = generate_questions_and_response(query)
    print(response.model_dump_json(indent=2))
    """
    {
      "follow_up_questions": [
        {
          "question": "When was superconductivity discovered?",
          "answer": "Superconductivity was discovered in 1911 by
          Heike Kamerlingh Onnes."
        },
        {
          "question": "Who was president of the U.S. in 1911?",
          "answer": "The president of the U.S. in 1911 was William
          Howard Taft."
        }
      ],
      "answer": "The president of the U.S. when superconductivity was
      discovered in 1911 was William Howard Taft."
    }
    """
```

### References

<sup id="ref-1">1</sup>: [Getting MoRE out of Mixture of Language Model Reasoning Experts](http://users.umiacs.umd.edu/~jbg/docs/2023_findings_more.pdf)
