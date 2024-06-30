---
title: "Step-Back Prompting"
description: "Step-back prompting is a two-step prompting technique that asks the LLM a step-back question to gather context for the query"
---

# Step-Back Prompting

Step-Back prompting<sup><a href="https://arxiv.org/abs/2310.06117">1</a></sup> is a variation of Chain of Thought prompting. The LLM is prompted in two steps:

1. **Abstraction**: Ask the LLM a generic, higher-level concept. This is generally topic-specific. This is known as the _step-back question_.
2. **Reasoning**: Ask the LLM the original question, given its answer to the abstract question. This is known as _abstracted-grounded reasoning_.

!!! example "Step-Back Prompting Example"

    **Original Question**: What happens to the pressure of an ideal gas when temperature and volume are increased?

    **Step-Back Prompt**: What are the physics concepts associated with this question?

    **Reasoning Prompt**: {step-back response} {original question}

This has been shown to improve scores on reasoning benchmarks for PaLM-2L and GPT-4.<sup><a href="https://arxiv.org/abs/2406.06608">\*</a></sup>

```python
import openai
import instructor
from pydantic import BaseModel
from typing import Iterable, Literal

client = instructor.from_openai(openai.OpenAI())

class Education(BaseModel):
    degree: Literal["Bachelors", "Masters", "PhD"]
    school: str
    topic: str
    year: int

class Response(BaseModel):
    school: str

# Step 1: Ask the stepback question
stepback = client.chat.completions.create(
    model="gpt-4o",
    response_model=Iterable[Education],
    messages=[
        {
            "role": "user",
            "content": "What was Estella Leopold’s education history?" # (1)!
        }
    ]
)

for education in stepback:
    print(education)
# >degree='Bachelors' school='University of Wisconsin' topic='Botany' year=1948
# >degree='Masters' school='University of California, Berkeley' topic='Paleobotany' year=1950
# >degree='PhD' school='Yale University' topic='Botany' year=1955

# Step 2: Answer the question with context from the stepback response
response = client.chat.completions.create(
    model="gpt-4o",
    response_model=Response,
    messages=[
        {
            "role": "user",
            "content": f"""
                {stepback}
                Estella Leopold went to which school between Aug 1954 and Nov 1954?
                """ # (2)!
        }
    ]
)

print(response.school)
# >Yale University
```

1. This is the stepback question.
2. This is the original question appended with context from the LLM's response to the stepback question.

### References

<sup id="ref-1">1</sup>: [Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models](https://arxiv.org/abs/2310.06117)

<sup id="ref-asterisk">\*</sup>: [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608)
