---
title: "Style Prompting"
description: "To contrain a model's response to fit the boundaries of our task, we can specify a style."
---

How can we constrain model outputs through prompting alone?

To contrain a model's response to fit the boundaries of our task, we can specify a style.

Stylistic constraints can include:
 
 - **writing style**: write a *flowery* poem
 - **tone**: write a *dramatic* poem
 - **mood**: write a *happy* poem
 - **genre**: write a *mystery* poem

## Implementation

```python hl_lines="21"
import instructor
from pydantic import BaseModel
import openai


class Email(BaseModel):
    subject: str
    message: str


client = instructor.from_openai(openai.OpenAI())

if __name__ == "__main__":
    email = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": """
                Write an email addressed to Mr. John Smith from Jane Doe.
                The email should be formal and the message should be polite and respectful.
                """,
            }
        ],
        response_model=Email,
    )

    print(email.subject)
    #> Formal Correspondence
    print(email.message)
    """
    Dear Mr. John Smith,

    I hope this email finds you well. My name is Jane Doe, and I am writing to you regarding [specific topic]. I would appreciate the opportunity to discuss this matter further with you at your earliest convenience.

    Thank you for your time and consideration. I look forward to your response.

    Best regards,
    Jane Doe
    """
```

## Stylistic Constraint Examples

| Constraint     | Possible Phrases                                                                  |
|----------------|-----------------------------------------------------------------------------------|
| Writing Style  | Functional, Flowery, Candid, Prosaic, Ornate, Poetic                              |
| Tone           | Dramatic, Humorous, Optimistic, Sad, Formal, Informal                             |
| Mood           | Angry, Fearful, Happy, Sad                                                        |
| Genre          | Historical Fiction, Literary Fiction, Science Fiction, Mystery, Dystopian, Horror |

!!! info "More Stylistic Constraints"

    To see even more examples of these stylistic constraints and additional constraints (**characterization**, **pacing**, and **plot**), check out this<sup><a href="https://arxiv.org/abs/2302.09185">1</a></sup> paper.

## References

<sup id="ref-1">1</sup>: [Bounding the Capabilities of Large Language Models in Open Text Generation with Prompt Constraints](https://arxiv.org/abs/2302.09185)

