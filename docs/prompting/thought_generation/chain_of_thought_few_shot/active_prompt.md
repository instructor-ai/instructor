---
title: "Active Prompting"
description: ""
---

# Active Prompting

Active prompting<sup><a href="https://arxiv.org/abs/2302.12246">1</a></sup> is a method used to identify the most effective examples for human annotation. It is particularly useful when dealing with large amounts of unlabeled data, as it provides a system to prioritze which data to label by humans. The process involves four key steps:

1. **Uncertainty Estimation**: Assessing the uncertainty of the model's predictions
2. **Selection**: Choosing the most uncertain examples for human annotation
3. **Annotation**: Having humans label the selected examples
4. **Inference**: Using the newly labeled data to improve the model's performance

## Uncertainty Estimation

In this step, we define an unsupervised method to measure the uncertainty of an LLM in answering a given example.

!!! example "Uncertainty Estimation Example"

    Let's say we ask an LLM the following query:
    >query = "Classify the sentiment of this sentence as positive or negative: I am very excited today."

    and the LLM returns:
    >response = "positive"

    The goal of uncertainty estimation is to answer: **How sure is the LLM in this response?**

In order to do this, we query the LLM with the same example _k_ times. Then, we use the _k_ responses to determine how dissimmilar these responses are. Three possible metrics<sup><a href="https://arxiv.org/abs/2302.12246">1</a></sup> are:

1. Disagreement
2. Entropy
3. Variance

Below is an example of uncertainty estimation for a single input example using the disagreement uncertainty metric.

```python
import instructor
from pydantic import BaseModel
from openai import OpenAI

class UserInfo(BaseModel):
    name: str
    age: int

client = instructor.from_openai(OpenAI())

k = 5 # (1)!
responses = []

# Query the LLM k times
for _ in range(k):
    user_info = client.chat.completions.create(
        model="gpt-4o",
        response_model=UserInfo,
        messages=[{"role": "user", "content": "How old is Jason Liu?"}],
    )
    responses.append(user_info.age)

# Calculate the disagreement
unique_responses = set(responses)
h = len(unique_responses)
u = h / k

for i, response in enumerate(responses):
    print(f"Response {i+1}: Age: {response}")
#>Response 1: Age: 25
#>Response 2: Age: 0
#>Response 3: Age: 30
#>Response 4: Age: 30
#>Response 5: Age: 30

print(f"\nDisagreement (u): {u}")
#>Disagreement (u): 0.6
```

1. _k_ is the number of times to query the LLM with a single unlabeled example

This process will then be repeated for all unlabeled examples.

!!! info "Uncertainty Estimation Output"

    Before uncertainty estimation, we have a list of unlabeled examples (List[x]).

    After uncertainty estimation, we have a list of unlabeled examples and their uncertainties (List[x: uncertainty]).

## Selection & Annotation

Once we have a set of examples and their uncertainties, we can select _n_ of them to be annotated by humans. Here, we choose the examples with the highest uncertainties.

## Inference

Now, each time the LLM is prompted, we can include the newly-annotated examples.

## References

<sup id="ref-1">1</sup>: [Active Prompting with Chain-of-Thought for Large Language Models](https://arxiv.org/abs/2302.12246)

<sup id="ref-asterisk">\*</sup>: [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608)
