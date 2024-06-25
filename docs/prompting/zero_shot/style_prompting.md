---
title: "Style Prompting"
description: "Style Prompting include a specific style, tone or genre in the prompt"
---

Style prompting<sup><a href="https://arxiv.org/abs/2302.09185">1</a></sup> includes a style, tone, or genre in the prompt. The effect can be similar to [role prompting](https://python.useinstructor.com/prompting/zero_shot/role_prompting/).<sup><a href="https://arxiv.org/abs/2406.06608">\*</a></sup>

```python
import instructor
from pydantic import BaseModel
import openai


class Email(BaseModel):
    message: str


client = instructor.from_openai(openai.OpenAI())


email = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": "Write an email addressed to Mr. John Smith from Jane Doe. \
            The email should be formal and the message should be polite\
            and respectful. The topic of the email is to invite Mr Smith\
            to a business meeting on July 15, 2024 at 10:00 AM in Room 123.",  # (!) !
        }
    ],
    response_model=Email,
)

print(email.message)
```

!!! note "Generated Email"
```
Dear Mr. John Smith,

    I hope this email finds you well. My name is Jane Doe, and I am writing to cordially invite you to attend a business meeting scheduled for July 15, 2024, at 10:00 AM in Room 123.

    Your presence at this meeting would be greatly appreciated, as your insights and expertise would be invaluable to our discussions.

    If you have any questions or require any additional information, please do not hesitate to contact me.

    Thank you for your time and consideration. I look forward to the possibility of your attendance.

    Best regards,
    Jane Doe
    ```

<sup id="ref-1">1</sup>: [Bounding the Capabilities of Large Language Models in Open Text Generation with Prompt Constraints](https://arxiv.org/abs/2302.09185)

<sup id="ref-asterisk">\*</sup>: [The Prompt Resport: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608)
