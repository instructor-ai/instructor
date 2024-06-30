---
title: "Tab-CoT: Zero-shot Tabular Chain of Thought"
description: "Tab-CoT encourages LLMs to output reasoning as a markdown table, improving the structure and reasoning of its output"
---

Tab-CoT <sup><a href="https://arxiv.org/pdf/2305.17812">1</a></sup> aims to help language models output their reasoning as a structured markdown table, thus improving the structure and reasoning of it's output.

!!! example "Tab-CoT sample prompt"

    {input prompt}

    | step | subquestion | procedure | result |

We can implement this in Instructor as a response object as seen below to ensure we get exactly the data that we want.

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field


client = instructor.from_openai(OpenAI())


class ReasoningStep(BaseModel):
    step: int = Field(..., description="This is the step number")
    subquestion: str = Field(..., description="This is the subquestion")
    procedure: str = Field(
        ...,
        description="This represents any intermediate computation that was done in the reasoning process. Leave empty if no computation is needed",
    )
    result: str = Field(..., description="This is the result\
         of the reasoning step")


class Response(BaseModel):
    reasoning: list[ReasoningStep] = Field(
        ...,
        description="This is a list of reasoning steps to get the answer",
    )
    answer: str = Field(..., description="This is the answer\
         to the user's question")


response = client.chat.completions.create(
    model="gpt-4o",
    response_model=Response,
    messages=[
        {
            "role": "system",
            "content": """
                The bakers at the Beverly Hills Bakery baked
                200 loaves of bread on Monday morning.
                They sold 93 loaves in the morning and 39
                loaves in the afternoon. A grocery store
                returned 6 unsold loaves. How many loaves
                of bread did they have left?
            """.strip(),
        },
    ],
)
```

This generates the following reasoning step and the correct response of 74.

| Step | Subquestion                                                                          | Procedure                                                                                    | Result                |
| ---- | ------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------- | --------------------- |
| 1    | How many loaves of bread were sold during the entire day?                            | Add the number of loaves sold in the morning and the number of loaves sold in the afternoon. | 93 + 39 = 132 loaves  |
| 2    | How many loaves of bread were left after sales of the day?                           | Subtract the total loaves sold from the total loaves baked.                                  | 200 - 132 = 68 loaves |
| 3    | How many loaves of bread were left after the grocery store returned 6 unsold loaves? | Add the number of loaves returned to the number of loaves left after sales.                  | 68 + 6 = 74 loaves    |

### References

<sup id="ref-1">1</sup>: [Tab-CoT: Zero-shot Tabular Chain of Thought](https://arxiv.org/pdf/2305.17812)
