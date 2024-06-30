---
title: "Style Prompting"
description: "Style Prompting include a specific style, tone or genre in the prompt"
---

## What is Style Prompting?

Style Prompting<sup><a href="https://arxiv.org/abs/2302.09185">1</a></sup> aims to influence the structure by prompting the LLM to adhere to a specific writing style. This includes things such as indicating a certain tone, pacing or even characterization to follow.

The effect can be similar to [role prompting](https://python.useinstructor.com/prompting/zero_shot/role_prompting/).<sup><a href="https://arxiv.org/abs/2406.06608">\*</a></sup>

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

??? note "Sample Style Prompting output"

    ```
    Dear Mr. John Smith,

    I hope this email finds you well. My name is Jane Doe, and I am writing to cordially invite you to attend a business meeting scheduled for July 15, 2024, at 10:00 AM in Room 123.

    Your presence at this meeting would be greatly appreciated, as your insights and expertise would be invaluable to our discussions.

    If you have any questions or require any additional information, please do not hesitate to contact me.

    Thank you for your time and consideration. I look forward to the possibility of your attendance.

    Best regards,
    Jane Doe
    ```

## Useful Tips

Here are some useful phrases that you might want to include when using style prompting to get the desired output that you want.

| Category         | Template                                                                                                                                                                                                                                                                                                     | Example Phrases                                                                                                                               |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Writing Style    | 1. Write a **style** passage about **topic**<br>2. Write a passage with a **style** writing style about **topic**<br>3. Write a **style** passage about **topic**                                                                                                                                            | **style**: functional, flowery, candid, prosaic, ornate, or poetic<br>**topic**: sunsets, strawberries, or writing a paper                    |
| Tone             | 1. Write a **tone** passage about **subject**<br>2. Write a passage about **subject** with a **tone** tone<br>3. Create a **tone** passage about **subject**                                                                                                                                                 | **tone**: dramatic, humorous, optimistic, sad<br>**subject**: love, life, humanity                                                            |
| Emotional Tone   | 1. Write a passage about **subject** that makes the reader feel **emotion**<br>2. Write a passage about **subject** with a **emotion** mood<br>3. Create a passage about **subject** that makes the reader feel **emotion**                                                                                  | **subject**: love, life, humanity<br>**emotion**: angry, fearful, happy, sad, envious, anxious, proud, regretful, surprised, loved, disgusted |
| Characterization | 1. Write a story about **subject** with indirect characterization<br>2. Write a story about **subject** with direct characterization<br>3. Create a story about **subject** where the characters are described indirectly<br>4. Create a story about **subject** where the characters are described directly | **subject**: lovers, cats, survivors                                                                                                          |
| Pacing           | 1. Write a **pace**-paced story about **subject**<br>2. Write a story about **subject** that is **pace**-paced<br>3. Create a **pace**-paced story about **subject**                                                                                                                                         | **pace**: fast, slow<br>**subject**: lovers, cats, survivors                                                                                  |

## References

<sup id="ref-1">1</sup>: [Bounding the Capabilities of Large Language Models in Open Text Generation with Prompt Constraints](https://arxiv.org/abs/2302.09185)

<sup id="ref-asterisk">\*</sup>: [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608)
