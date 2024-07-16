---
title: "Select Effective Examples"
description: "KNN can be leveraged to choose the most effective examples to use for a given query."
---

How can we select effective in-context examples?

Given a pool of possible in-context examples, KNN can be leveraged to choose the most effective examples to use for a given query.

In the below implementation using `instructor`, we follow these steps:

1. Embed the possible examples
2. Embed the query
3. Find the *k* examples closest to the query
4. Use the chosen examples as the context for the LLM

```python
import instructor
from pydantic import BaseModel
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import numpy as np


class Example(BaseModel):
    question: str
    answer: str


class Response(BaseModel):
    answer: str


client = instructor.from_openai(OpenAI())
model = SentenceTransformer("all-distilroberta-v1")


def distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))  # Cosine distance


def knn(embedded_examples, examples, embedded_query, k=2):
    distances = [distance(embedded_query, ex) for ex in embedded_examples]
    sorted_indicies = np.argsort(distances)
    return [examples[i] for i in sorted_indicies[:k]]


if __name__ == "__main__":
    examples = [
        Example(question="What is the capital of France?", answer="Paris"),
        Example(question="Who wrote Romeo and Juliet?", answer="Shakespeare"),
        Example(question="What is the largest planet in our solar system?", answer="Jupiter"),
        Example(question="What is the capital of Germany?", answer="Berlin"),
    ]
    query = "What is the capital of Italy?"

    # Step 1: Embed the examples
    embedded_examples = [model.encode(example.question) for example in examples]

    # Step 2: Embed the query
    embedded_query = model.encode(query)

    # Step 3: Find the k closest examples to the query
    k_closest_examples = knn(embedded_examples, examples, embedded_query)

    for example in k_closest_examples:
        print(example)
        #> question='What is the capital of France?' answer='Paris'
        #> question='What is the capital of Germany?' answer='Berlin'

    # Step 4: Use these examples as in-context examples
    response = client.chat.completions.create(
        model="gpt-4o",
        response_model=Response,
        messages=[
            {
                "role": "user",
                "content": f"""
                           {k_closest_examples}
                           {query}
                           """,
            }
        ],
    )

    print(response.answer)
    #> Rome
```

### References

<sup id="ref-1">1</sup>: [What Makes Good In-Context Examples for GPT-3?](https://arxiv.org/abs/2101.06804)

<sup id="ref-asterisk">\*</sup>: [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608)
