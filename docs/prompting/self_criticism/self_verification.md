---
description: "Self Verification involves getting language models to generate a candidate response before evaluating each individual intermediate reasoning step to verify if it's logical entailment holds"
---

We can verify the correctness of the reasoning steps taken by our Large Language Model by rewriting them as logical entailments. This enables us to use an LLM to check if the original statement can be derived from the new logical entailment. By doing this, we can score each reasoning step and obtain a metric for the quality of the response.

We can scale this out to multiple candidate solutions to choose the best solution.

```python hl_lines="21-24 34-38 56-83 95-98"
import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
import asyncio

client = instructor.from_openai(AsyncOpenAI())


class SelfVerification(BaseModel):
    chain_of_thought: str
    is_valid_reasoning_step: bool


class RewrittenValidationStep(BaseModel):
    chain_of_thought: str
    rewritten_reasoning_step: str


class Response(BaseModel):
    reasoning_steps: list[str] = Field(
        description="""Logic reasoning steps that allow
        us to arrive at the final answer. Make sure to
        include specific figures and calculations that
        allow you to derive the final correct answer""",
    )
    correct_answer: str


def generate_reasoning_and_response(query: str):
    return client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """You are an expert AI question
                answering system. Make sure to generate a
                list of reasoning steps that are consistent
                and logical before generating your final
                response.""",
            },
            {
                "role": "user",
                "content": query,
            },
        ],
        response_model=Response,
        model="gpt-4o",
    )


async def evaluate_reasoning_step(reasoning_step: str):
    rewritten_reasoning_step = await client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """
                You are an expert AI Rewritter. You are
                about to be passed a reasoning step and
                your goal is to rewrite it so that we can
                verify if the final conclusion can be
                obtained from the conclusion.

                Here are some examples

                Example 1
                Reasoning Step: Jackie has 10 apples so
                Jackie has 10-8=2 more apples than Adam.
                So the answer is 2.
                Rewritten Reasoning Step: Jackie has X
                apples. Adam has 8 apples. Jackie has 2
                more apples than Adam. Therefore Jackie
                must have had 10 apples at the start.
                Therefore X must be 10.

                Example 2
                Reasoning Step: John reads 4 books a day.
                He reads every Monday and Tuesday which
                are 2 days per week. Therefore he reads 8
                books a week.
                Rewritten Reasoning Step: John reads X
                books a day. He reads every Monday and
                Tuesday which are 2 days per week.
                Therefore he reads 8 books a week.
                Therefore X must be 4.
                """,
            },
            {"role": "user", "content": reasoning_step},
        ],
        response_model=RewrittenValidationStep,
        model="gpt-4o",
    )
    return await client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """You are an expert AI Statement
                Verification tool. You are about to be
                passed a logical step, and asked to verify
                if it's a valid reasoning step or not.""",
            },
            {
                "role": "user",
                "content": rewritten_reasoning_step.rewritten_reasoning_step,
            },
        ],
        response_model=SelfVerification,
        model="gpt-4o",
    )


async def evaluate_model_reasoning(reasoning: list[str]):
    tasks = [evaluate_reasoning_step(step) for step in reasoning]
    results = await asyncio.gather(*tasks)
    return results


if __name__ == "__main__":
    query = """
    Tim wanted to make lemonade for a pool party. For a
    gallon of lemonade, his recipe called for 1 cup of
    fresh lemon juice. He found that 6 lemons would
    yield 1 cup of juice. He figured he would need to
    make 4 gallons of lemonade for the party. His best
    friend Allen asked if Tim could make an extra
    gallon for him that was twice as tart as the other
    gallons. How many lemons will Tim need?
    """
    response = asyncio.run(generate_reasoning_and_response(query))

    for step in response.reasoning_steps:
        print(step)
        """
        To calculate the number of lemons needed, first
        determine the lemons required for the primary
        lemonade. For 4 gallons, 4 cups of lemon juice
        are needed, equating to 4 * 6 = 24 lemons.
        """
        """
        For the extra gallon of tart lemonade, which is
        twice as tart, it requires 2 cups of lemon
        juice. This equates to 2 * 6 = 12 lemons.
        """
        """
        Summing these amounts, Tim needs 24 + 12 = 36
        lemons in total.
        """

    reasoning_evaluation = asyncio.run(
        evaluate_model_reasoning(response.reasoning_steps)
    )

    valid_reasoning_count = 0
    for reasoning in reasoning_evaluation:
        if reasoning.is_valid_reasoning_step:
            valid_reasoning_count += 1

    print(valid_reasoning_count/len(reasoning_evaluation))
    #> 1
    print(response.correct_answer)
    #> 36
```
