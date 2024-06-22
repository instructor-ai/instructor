---
title: "Emotion Prompting"
description: ""
keywords: "prompt, prompting, prompt engineering, llm, gpt, openai, emotion prompting, model, AI, python, instructor, zero-shot"
---

Emotion prompting<sup><a href="https://arxiv.org/abs/2307.11760">1</a></sup> uses phrases that have psychological relevance to humans in the LLM prompt. This may lead to improved performance on benchmarks and open-ended text generation.<sup><a href="https://arxiv.org/abs/2406.06608">\*</a></sup>

```python
import openai
import instructor
from pydantic import BaseModel

class Recipe(BaseModel):
    dish_name: str
    ingredients: list[str]
    steps: list[str]

client = instructor.from_openai(openai.OpenAI())

emotion_prompt = """
    I want to impress my friends with my cooking.
    Provide a recipe for a delicious chocolate cake.
    """

recipe = client.chat.completions.create(
    model="gpt-4o",
    response_model=Recipe,
    messages=[{"role": "user", "content": emotion_prompt}],
)

print("Dish Name:", recipe.dish_name)
print("Ingredients:")
for ingredient in recipe.ingredients:
    print("-", ingredient)
print("Steps:")
for step in recipe.steps:
    print("-", step)
```

<sup id="ref-1">1</sup>: [Large Language Models Understand and Can be Enhanced by Emotional Stimuli](https://arxiv.org/abs/2307.11760)

<sup id="ref-asterisk">\*</sup>: [The Prompt Resport: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608)
