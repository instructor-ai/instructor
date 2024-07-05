---
title: "Recall From Memory"
description: "Memory-of-Thought (MoT) is a two-step framework that helps an LLM improve its responses through self-thinking and memory."
---

How do we improve an LLM's reasoning without new data or parameter updates?

Memory-of-Thought (MoT) is a two-step framework that helps an LLM improve its responses through self-thinking and memory, mimicing humans. The steps are:

1. **Before the query**: 
    1. the LLM *pre-thinks* on an unlabeled dataset and
    2. saves the highest confidence thoughts as external memory
2. **During the query**: the LLM recalls relevant memory to help answer the query

MoT has shown to improve ChatGPT's reasoning abilities and improve results from other CoT methods<sup><a href="https://arxiv.org/abs/2305.05181">1</a></sup>.

## Before the Query

First, **the model *pre-thinks***. For each example, we generate a CoT line of reasoning *n* times. 

Then, we **filter out low-confidence thoughts**. If the variability of these answers is high, we exclude the example. If not, we use the most consistent answer from the n generated answers. Variability here is implemented as entropy<sup><a href="https://arxiv.org/abs/2305.05181">1</a></sup>.

We can implement this using `instructor` as seen below:

```python
import instructor
from pydantic import BaseModel
from openai import OpenAI
import math
from collections import Counter

client = instructor.from_openai(OpenAI())

n = 3  # Number of reasonings to query for each example


class CoTReasoning(BaseModel):
    question: str
    reasoning_steps: list[str]
    answer: str


# Generates 1 reasoning path for 1 example
def cot_query(example):
    return client.chat.completions.create(
        model="gpt-4o",
        response_model=CoTReasoning,
        temperature=0.1,
        messages=[{"role": "user", "content": f"Think step by step: {example}"}], # (1) !
    )


def calculate_entropy(reasonings):
    answers = [reasoning.answer for reasoning in reasonings]
    unique_answers = set(answers)
    probabilities = [answers.count(answer) / len(answers) for answer in unique_answers]
    return -sum(p * math.log(p) for p in probabilities)  # Entropy calculation


def most_consistent(reasonings):
    answers = [reasoning.answer for reasoning in reasonings]
    most_common_answer = Counter(answers).most_common(1)[0][0]
    for reasoning in reasonings:
        if reasoning.answer == most_common_answer:
            return reasoning


def think(example, threshold=0.5):
    reasonings = [cot_query(example) for _ in range(n)]
    # Choose most consistent answer if entropy is low enough, else don't include example
    return (
        most_consistent(reasonings)
        if calculate_entropy(reasonings) <= threshold
        else None
    )


if __name__ == "__main__":
    dataset = [
        "A car travels 150 miles in 3 hours. If it continues at the same speed, how far will it travel in 7 hours?",
        "A store sells a pack of 12 pens for $3.60. How much would 20 pens cost at the same rate?",
        "A tank can hold 300 gallons of water. If it is filled at a rate of 50 gallons per hour, how long will it take to fill the tank?",
        "A recipe requires 2.5 cups of flour to make 15 cookies. How much flour is needed to make 40 cookies?",
        "A cyclist covers a distance of 120 miles in 4 hours. What is the average speed in miles per hour, and how long will it take to cover 300 miles at the same speed?",
    ]

    # Prethink
    thoughts = [think(example) for example in dataset]
    thoughts = [
        thought for thought in thoughts if thought is not None
    ]  # Remove None values

    print(f"{len(dataset)-len(thoughts)} example(s) filtered out during prethink")
    #> 1 example(s) filtered out during prethink

    for thought in thoughts:
        print(thought)
        """
        question='A car travels 150 miles in 3 hours. If it continues at the same speed, how far will it travel in 7 hours?' reasoning_steps=['First, calculate the speed of the car by dividing the distance traveled by the time taken.', 'Speed = 150 miles / 3 hours = 50 miles per hour.', 'Next, use the speed to calculate the distance the car will travel in 7 hours.', 'Distance = Speed * Time = 50 miles per hour * 7 hours = 350 miles.'] answer='The car will travel 350 miles in 7 hours.'
        """
        """
        question='A tank can hold 300 gallons of water. If it is filled at a rate of 50 gallons per hour, how long will it take to fill the tank?' reasoning_steps=['Determine the total capacity of the tank.', 'Identify the rate at which the tank is being filled.', 'Divide the total capacity by the filling rate to find the time required.'] answer='It will take 6 hours to fill the tank.'
        """
        """
        question='A recipe requires 2.5 cups of flour to make 15 cookies. How much flour is needed to make 40 cookies?' reasoning_steps=['Determine the amount of flour needed for one cookie by dividing the total flour by the number of cookies.', 'Multiply the amount of flour needed for one cookie by the desired number of cookies.'] answer='6.67 cups of flour'
        """
        """
        question='A cyclist covers a distance of 120 miles in 4 hours. What is the average speed in miles per hour, and how long will it take to cover 300 miles at the same speed?' reasoning_steps=['First, calculate the average speed by dividing the total distance by the total time.', 'Next, use the average speed to determine the time required to cover 300 miles by dividing 300 miles by the average speed.'] answer='The average speed is 30 miles per hour, and it will take 10 hours to cover 300 miles at the same speed.'
        """
```

1. This can also be implemented using few-shot CoT<sup><a href="https://arxiv.org/abs/2305.05181">1</a></sup>

## During the Query
[wip]

## References

<sup id="ref-1">1</sup>: [MoT: Memory-of-Thought Enables ChatGPT to Self-Improve](https://arxiv.org/abs/2305.05181)

<sup id="ref-asterisk">\*</sup>: [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608)