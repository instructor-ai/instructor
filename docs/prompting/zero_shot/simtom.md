---
title: "SimToM (Simulated Theory of Mind)"
description: "SimToM (Simulated Theory of Mind) is a two-step prompting technique that focuses on improving the LLM's handling of complex questions involving multiple people or objects"
---

# SimToM (Simulated Theory of Mind)

SimToM (Simulated Theory of Mind)<sup><a href="https://arxiv.org/abs/2311.10227">1</a></sup> is a prompting technique designed to handle complex questions involving multiple people or objects. It involves two steps:

1. Establish the set of facts known to a specific person or entity
2. Answer the question based on solely on those established facts

!!! example "SimToM Example Template"

    **Step 1**: Given the following context, list the facts that {entity} would know. Context: {context}

    **Step 2**: You are {entity}. Answer the following question based only on these facts you know {facts}. Question: {question}

This approach can help eliminate the influence of irrelevant information in the prompt.<sup><a href="https://arxiv.org/abs/2406.06608">\*</a></sup>

```python
import openai
import instructor
from pydantic import BaseModel
from typing import Iterable

client = instructor.from_openai(openai.OpenAI())

class KnownFact(BaseModel):
    fact: str

class Response(BaseModel):
    location: str

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
                    Alice puts the book on the table.
                    Alice leaves the room.
                    Bob moves the book to the shelf.
                    Where does {entity} think the book is?

                    List only the facts relevant to {entity}.
                    """
            }
        ]
    )

    # Step 2: Answer the question based on known facts
    response = client.chat.completions.create(
        model="gpt-4o",
        response_model=Response,
        messages=[
            {
                "role": "system",
                "content": f"You are {entity}. Answer the following question based only on these facts you know: {" ".join([str(fact) for fact in known_facts])}"
            },
            {
                "role": "user",
                "content": f"Question: Where does {entity} think the book is?"
            }
        ]
    )

    for fact in known_facts:
        print(fact)
    print(response.location)

simtom("alice")
# >fact='Alice puts the book on the table.'
# >fact='Alice leaves the room.'
# >on the table
simtom("bob")
# >fact='Alice puts the book on the table.'
# >fact='Alice leaves the room.'
# >fact='Bob moves the book to the shelf.'
# >shelf
```

### References

<sup id="ref-1">1</sup>: [Think Twice: Perspective-Taking Improves Large Language Models' Theory-of-Mind Capabilities](https://arxiv.org/abs/2311.10227)

<sup id="ref-asterisk">\*</sup>: [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608)
