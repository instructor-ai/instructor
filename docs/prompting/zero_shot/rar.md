---
title: "Rephrase and Respond"
description: "Rephrase and Respond aims to reduce prompt ambiguity and align the question more closely with the LLM's existing frame"
---

# Rephrase and Respond

Rephrase and Respond(RaR)<sup><a href="https://arxiv.org/pdf/2311.04205">1</a></sup> aims to reduce prompt ambiguity and align the question more closely with the LLM.

This could look something like this

!!! example "Rephrase and Respond Example"

    **User**: Take the last letters of the words in 'Edgar Bob' and concatenate them.

    **Rephrased Question**: Could you please form a new string or series of characters by joining together the final letters from each word in the phrase "Edgar Bob"?

    **Assistant**: The last letters in the words "Edgar" and "Bob" are "r" and "b", hence when concatenated, it forms "rb".

We can implement this in instructor as seen below.

```python
from pydantic import BaseModel, Field
import instructor
from openai import OpenAI

client = instructor.from_openai(OpenAI())


class ImprovedQuestion(BaseModel):
    rewritten_question: str = Field(
        ..., description="An improved, more specific version of the original question"
    )


class FinalResponse(BaseModel):
    response: str = Field(..., description="This is a response to an object")


def rewrite_question(question: str):
    return client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are an expert at rewriting questions to be more specific and answerable.",
            },
            {"role": "user", "content": f"The question is {question}"},
        ],
        response_model=ImprovedQuestion,
    )


def answer_question(question: str):
    return client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": question}],
        max_tokens=1000,
        response_model=FinalResponse,
    ).response


rewritten_query = rewrite_question(
    "Take the last letters of the words in 'Elon Musk' and concatenate them"
)
# What are the last letters of the words in 'Elon Musk', and
# what do you get if you concatenate them?

response = answer_question(rewritten_query.rewritten_question)
# The last letters of the words in 'Elon Musk' are 'n' and 'k'.
# If you concatenate them, you get 'nk'.
```

We can also achieve the same benefits by **using a better model to generate the question** before we prompt a weaker model - this is known as a two-step RaR.

## Useful Tips

Here are some phrases that you can add to your prompt to refine the question before you generate a response

- Reword and elaborate on the inquiry, then provide an answer.
- Reframe the question with additional context and detail, then provide an answer.
- Modify the original question for clarity and detail, then offer an answer.
- Restate and elaborate on the inquiry before proceeding with a response.
- Given the above question, rephrase and expand it to help you do better answering. Maintain all information in the original question.

### References

<sup id="ref-1">1</sup>: [Rephrase and Respond: Let Large Language Models Ask Better Questions for Themselves](https://arxiv.org/pdf/2311.04205)
