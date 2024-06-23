---
title: "Style Prompting"
description: ""
keywords: "prompt, prompting, prompt engineering, llm, gpt, openai, style prompting, model, AI, python, instructor, zero-shot, role prompting"
---

Style prompting<sup><a href="https://arxiv.org/abs/2302.09185">1</a></sup> includes a style, tone, or genre in the prompt. The effect can be similar to [role prompting](https://python.useinstructor.com/prompting/zero_shot/role_prompting/).<sup><a href="https://arxiv.org/abs/2406.06608">\*</a></sup>

```python
import instructor
from pydantic import BaseModel
import openai

class Email(BaseModel):
    recipient: str
    sender: str
    message: str

client = instructor.from_openai(openai.OpenAI())

style_prompt = {
    "role": "user",
    "content": (
        "Write a email addressed to Mr. John Smith from Jane Doe."
        "The email should be formal and the message should be polite and respectful." # Style prompting
        "Invite Mr. Smith to a business meeting on July 15, 2024 at 10:00 AM in Room 123."
    )
}

email = client.chat.completions.create(
    model="gpt-4o",
    messages=[style_prompt],
    response_model=Email,
)

print(f"Recipient: {email.recipient}")
print(f"Sender: {email.sender}")
print(f"Message: {email.message}")
```

<sup id="ref-1">1</sup>: [Bounding the Capabilities of Large Language Models in Open Text Generation with Prompt Constraints](https://arxiv.org/abs/2302.09185)

<sup id="ref-asterisk">\*</sup>: [The Prompt Resport: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608)
