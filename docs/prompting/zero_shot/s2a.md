---
title: "System 2 Attention (S2A)"
description: "System 2 Attention (S2A) is a two-step prompting technique that focuses on improving the LLM's attention to relevant information"
---

# System 2 Attention (S2A)

System 2 Attention (S2A)<sup><a href="https://arxiv.org/abs/2311.11829">1</a></sup> is a two-step prompting technique that aims to improve the LLM's focus on relevant information. The two steps are:

1. Asking the LLM to rewrite the prompt, removing any information unrelated to the question.
2. Passing this new, focused prompt to the LLM to generate the final response.

This technique can result in more factual and less opinionated LLM outputs<sup><a href="https://arxiv.org/abs/2311.11829">1</a></sup>.<sup><a href="https://arxiv.org/abs/2406.06608">\*</a></sup>

```python
import openai
import instructor
from pydantic import BaseModel, Field

client = instructor.from_openai(openai.OpenAI())

class Step1(BaseModel):
    relevant_context: str = Field(..., description="Relevant context")
    user_query: str = Field(..., description="The question from the user")

class Step2(BaseModel):
    answer: int

# Step 1: Rewrite the prompt
rewritten_prompt = client.chat.completions.create(
    model="gpt-4o",
    response_model=Step1,
    messages=[
        {
            "role": "user",
            "content": f"""
                Given the following text by a user, extract the part that is actually relevant to their question.
                Please include the actual question or query that the user is asking.

                Text by user:
                Mary has 3 times as much candy as Megan.
                Mary then adds 10 more pieces of candy to her collection.
                Max has 1000 more books than Mary.
                If Megan has 5 pieces of candy, how many does Mary have in total?
                """
        }
    ]
)

print(rewritten_prompt.relevant_context)
# >Mary has 3 times as much candy as Megan. Mary then adds 10 more pieces of candy to her collection.
print(rewritten_prompt.user_query)
# >If Megan has 5 pieces of candy, how many does Mary have in total?

# Step 2: Generate the final response using the rewritten prompt
final_response = client.chat.completions.create(
    model="gpt-4o",
    response_model=Step2,
    messages=[
        {
            "role": "user",
            "content": f"""{rewritten_prompt.relevant_context}
                Question: {rewritten_prompt.user_query}"""
        }
    ]
)

print(final_response.answer)
# >25
```

### References

<sup id="ref-1">1</sup>: [System 2 Attention (is something you might need too)](https://arxiv.org/abs/2311.11829)

<sup id="ref-asterisk">\*</sup>: [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608)
