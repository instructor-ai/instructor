---
title: "Role Prompting"
description: ""
keywords: "prompt, prompting, prompt engineering, llm, gpt, openai, role prompting, persona prompting, model, AI, python, instructor, zero-shot"
---

Role prompting<sup><a href="https://arxiv.org/abs/2310.00746">1</a>, <a href="https://arxiv.org/abs/2311.10054">2</a></sup>, or persona prompting<sup><a href="#ref-3">3</a>, <a href="https://arxiv.org/abs/2307.05300">4</a></sup>, assigns a specific role to the model in the prompt ("You are ..."). This can improve outputs for open-ended tasks<sup><a href="https://dl.acm.org/doi/10.1145/3411763.3451760">5</a></sup> and improve accuracy on benchmarks<sup><a href="https://arxiv.org/abs/2311.10054">2</a></sup>.<sup><a href="https://arxiv.org/abs/2406.06608">\*</a></sup>

```python
import openai
import instructor

from typing import List, Literal
from pydantic import BaseModel, Field

# Apply the patch to the OpenAI client
# enables response_model keyword
client = instructor.from_openai(openai.OpenAI())

LABELS = Literal["ACCOUNT", "BILLING", "GENERAL_QUERY"]


class MultiClassPrediction(BaseModel):
    labels: List[LABELS] = Field(
        ...,
        description="Only select the labels that apply to the support ticket.",
    )


def multi_classify(data: str) -> MultiClassPrediction:
    return client.chat.completions.create(
        model="gpt-4o",
        response_model=MultiClassPrediction,
        messages=[
            {
                "role": "system",
                "content": f"You are a support agent at a tech company. Only select the labels that apply to the support ticket.",
            },
            {
                "role": "user",
                "content": f"Classify the following support ticket: {data}",
            },
        ],
    )  # type: ignore


if __name__ == "__main__":
    ticket = "My account is locked and I can't access my billing info."
    prediction = multi_classify(ticket)
    assert {"ACCOUNT", "BILLING"} == {label for label in prediction.labels}
    print("input:", ticket)
    #> input: My account is locked and I can't access my billing info.
    print("labels:", LABELS)
    #> labels: typing.Literal['ACCOUNT', 'BILLING', 'GENERAL_QUERY']
    print("prediction:", prediction)
    #> prediction: labels=['ACCOUNT', 'BILLING']
```

<sup id="ref-1">1</sup>: [RoleLLM: Benchmarking, Eliciting, and Enhancing Role-Playing Abilities of Large Lanuage Models](https://arxiv.org/abs/2310.00746)  
<sup id="ref-2">2</sup>: [Is "A Helpful Assistant" the Best Role for Large Language Models? A Systematic Evaluation of Social Roles in System Prompts ](https://arxiv.org/abs/2311.10054)  
<sup id="ref-3">3</sup>: Douglas C. Schmidt, Jesse Spencer-Smith, Quchen Fu, and Jules White. 2023. Cataloging prompt patterns to enhance the discipline of prompt engineering. Dept. of Computer Science, Vanderbilt University.  
<sup id="ref-4">4</sup>: [Unleashing the Emergent Cognitive Synergy in Large Lanuage Models: A Task-Solving Agent through Multi-Persona Self-Collaboration ](https://arxiv.org/abs/2307.05300)  
<sup id="ref-5">5</sup>: [Prompt Programming for Large Language Models: Beyond the Few-Shot Paradigm](https://dl.acm.org/doi/10.1145/3411763.3451760)

<sup id="ref-asterisk">\*</sup>: [The Prompt Resport: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608)
