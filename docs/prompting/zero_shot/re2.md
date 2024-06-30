---
title: "Re-Reading Improves Reasoning in Large Language Models"
description: "We can see a small improvement of <4% in different models by just appending the phrase - Read The Question Again."
---

# Re-Reading improves reasoning in large languge models.

Re-Reading<sup><a href="https://arxiv.org/pdf/2309.06275">1</a></sup> aims to help enhance the reasoning capabilities of Large Language Models. We can do so by using the simple phrase

This could look something like this

!!! example "Re-Reading Template"

    **[ Input Query ]**
    Read the question again: **[ Input Query ]**

    **[ Critical Thinking Prompt  ]**

We can implement this in Instructor pretty simply.

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field


client = instructor.from_openai(OpenAI())


class Solution(BaseModel):
    final_answer: int = Field(..., description="This is the final answer")


question = """Roger has 5 tennis balls. He buys 2 more cans of tennis
balls. Each can has 3 tennis balls. How many tennis balls
does he have now?"""


response = client.chat.completions.create(
    model="gpt-4o",
    response_model=Solution,
    messages=[
        {
            "role": "system",
            "content": f"{question}. Read the question again.\n {question}. Adhere to the provided format when responding to the problem and make sure to think through this step by step.",
        },
    ],
)

print(response.final_answer) # 11
```

### References

<sup id="ref-1">1</sup>: [Re-Reading Improves Reasoning in Large Language Models](https://arxiv.org/pdf/2309.06275)
