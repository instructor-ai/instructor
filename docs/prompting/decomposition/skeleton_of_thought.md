---
title: "Generate in Parallel"
description: "Skelelton-of-Thought is a technique which prompts an LLM to generate a skeleton outline of the response, then completes each point in the skeleton in parallel."
---

How do we decrease the latency of an LLM pipeline?

Skelelton-of-Thought is a technique which prompts an LLM to generate a skeleton outline of the response, then completes each point in the skeleton in parallel. The parallelism can be achieved by parallel API calls or batched decoding.

Below is an example of an implementation using parallel API calls with `instructor`:

```python
import instructor
from pydantic import BaseModel
from openai import OpenAI
# from multiprocessing import Pool

client = instructor.from_openai(OpenAI())


class Point(BaseModel):
    index: int
    description: str


class Skeleton(BaseModel):
    points: list[Point]


class Response(BaseModel):
    response: str


def get_skeleton(question):
    return client.chat.completions.create(
        model="gpt-4o",
        response_model=Skeleton,
        messages=[
            {
                "role": "user",
                "content": f"""
                You’re an organizer responsible for only giving the skeleton (not the full content) for answering the question.
                Provide the skeleton in a list of points (numbered 1., 2., 3., etc.) to answer the question.
                Instead of writing a full sentence, each skeleton point should be very short with only 3∼5 words.
                Generally, the skeleton should have 3∼10 points.

                Now, please provide the skeleton for the following question.

                <question>
                {question}
                </question>

                Skeleton:
                """,
            }
        ],
    )


def expand_point(input):
    question, skeleton, point_index = input

    return client.chat.completions.create(
        model="gpt-4o",
        response_model=Response,
        messages=[
            {
                "role": "user",
                "content": f"""
                You’re responsible for continuing the writing of one and only one point in the overall answer to the following question.

                <question>
                {question}
                </question>

                The skeleton of the answer is:

                <skeleton>
                {skeleton}
                </skeleton>

                Continue and only continue the writing of point {point_index}.
                Write it **very shortly** in 1∼2 sentence and do not continue with other points!
                """,
            }
        ],
    ).response


if __name__ == "__main__":
    query = "Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions."

    # Step 1: Get the skeleton
    skeleton = get_skeleton(query)

    for point in skeleton.points:
        print(point)
        #> index=1 description='Introduction to Hawaii trip'
        #> index=2 description='First impressions'
        #> index=3 description='Cultural insights'
        #> index=4 description='Traditional food tasting'
        #> index=5 description='Visit to historical sites'
        #> index=6 description='Beach experiences'
        #> index=7 description='Adventure activities'
        #> index=8 description='Local festivals and events'
        #> index=9 description='Meeting locals'
        #> index=10 description='Conclusion and reflections'

    # Step 2: Expand on each point in parallel
    # with Pool() as pool:
    #     args_list = [
    #         [query, skeleton.model_dump_json(), point.index]
    #         for point in skeleton.points
    #     ]
    #     results = pool.map(expand_point, args_list)

    # for point, result in zip(skeleton.points, results):
    #     print(f"Point {point.index}: {result}")
```


### References

<sup id="ref-1">1</sup>: [Skeleton-of-Thought: Prompting LLMs for Efficient Parallel Generation](https://arxiv.org/abs/2307.15337)

<sup id="ref-asterisk">\*</sup>: [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608)
