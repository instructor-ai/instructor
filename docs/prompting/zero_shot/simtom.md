---
title: "SimToM (Simulated Theory of Mind)"
description: "SimToM (Simulated Theory of Mind) is a two-step prompting technique that focuses on improving the LLM's handling of complex questions involving multiple people or objects"
---

# SimToM (Simulated Theory of Mind)

SimToM (Simulated Theory of Mind)<sup><a href="https://arxiv.org/abs/2311.10227">1</a></sup> is a prompting technique designed to handle complex questions involving multiple people or objects. It involves two steps:

1. Establish the set of facts known to a specific person or entity
   > Given the following context, list the facts that {entity} would know. Context: {context}.
2. Answer the question based on solely on those established facts
   > You are {entity}. Answer the following question based only on these facts you know {facts}. Question: {question}

This approach can help eliminate the influence of irrelevant information in the prompt.<sup><a href="https://arxiv.org/abs/2406.06608">\*</a></sup>

```python
import openai
import instructor
from pydantic import BaseModel
from typing import Iterable

client = instructor.from_openai(openai.OpenAI())

class KnownFact(BaseModel):
    fact: str

class Ingredient(BaseModel):
    ingredient: str

def simtom(entity):
    # Step 1: Establish known facts
    known_facts = client.chat.completions.create(
        model="gpt-4o",
        response_model=Iterable[KnownFact],
        messages=[
            {
                "role": "user",
                "content": f"""Given the following context, list the facts that {entity} would know:

                    Context:
                    The woman in green fills a cup with steamed whole milk.
                    Then, the women does not see the man in purple replace the whole milk in the cup with oat milk.
                    The woman then adds cinnamon spice on top, which the man does not see.
                    Then, both observe that the drink is given to a customer.
                    What are the ingredients of the drink according to {entity}?

                    List only the facts relevant to {entity}:
                    """
            }
        ]
    )

    # Step 2: Answer the question based on known facts
    ingredients = client.chat.completions.create(
        model="gpt-4o",
        response_model=Iterable[Ingredient],
        messages=[
            {
                "role": "system",
                "content": f"You are {entity}. Answer the following question based only on these facts you know: {" ".join([str(fact) for fact in known_facts])}"
            },
            {
                "role": "user",
                "content": f"Question: What are the ingredients of the drink according to {entity}?"
            }
        ]
    )

    for fact in known_facts:
        print(fact)
    for ingredient in ingredients:
        print(ingredient)

simtom("the woman in green")
# >fact='The woman in green fills a cup with steamed whole milk.'
# >fact='The woman in green adds cinnamon spice on top.'
# >fact='The woman in green observes that the drink is given to a customer.'
# >ingredient='steamed whole milk'
# >ingredient='cinnamon spice'
simtom("the man in purple")
# >fact='The woman in green fills a cup with steamed whole milk.'
# >fact='The man in purple replaces the whole milk in the cup with oat milk.'
# >fact='Both observe that the drink is given to a customer.'
# >ingredient='oat milk'
```

### References

<sup id="ref-1">1</sup>: [Think Twice: Perspective-Taking Improves Large Language Models' Theory-of-Mind Capabilities](https://arxiv.org/abs/2311.10227)

<sup id="ref-asterisk">\*</sup>: [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608)
