---
description: "Meta Chain Of Thought involves decomposing an initial query into multiple sub questions. We then aggregate the response from each of these chains as context before prompting another LLM to generate a response"
---

Meta Chain Of Thought (Meta COT) <sup><a href="https://arxiv.org/pdf/2304.13007">1</a></sup>. involves the use of multiple reasoning chains to generate a response to a given query. This helps our model evaluate multiple potential reasoning paths and from there, determine a more accurate answer.

We can implement this using `instructor` as seen below.

```python hl_lines="28-29 44-45 74-77"
import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
import asyncio

client = instructor.from_openai(AsyncOpenAI())


class ReasoningAndResponse(BaseModel):
    intermdiate_reasoning: str = Field(description="""
    Intermediate reasoning steps""")
    correct_answer: str


class QueryDecomposition(BaseModel):
    queries: list[str] = Field(
        description="""A list of queries that need to be
        answered in order to derive the final answer"""
    )


async def generate_queries(query: str):
    return await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """You are a helpful assistant that
                decomposes a query into multiple queries.""",
            },
            {"role": "user", "content": query},
        ],
        response_model=QueryDecomposition,
    )


async def generate_reasoning_chain(query: str):
    return await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """
                Given a question and a context,
                answer the question step-by-step.

                If you are unsure, answer Unknown.
                """,
            },
            {"role": "user", "content": query},
        ],
        response_model=ReasoningAndResponse,
    )


async def batch_reasoning_chains(queries: list[str]):
    coros = [generate_reasoning_chain(query) for query in queries]
    results = await asyncio.gather(*coros)
    return results


async def generate_response(query: str, context: list[ReasoningAndResponse]):
    formatted_context = "\n".join(
        [f"{item.intermdiate_reasoning}\n{item.correct_answer}"
         for item in context]
    )

    return await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """
                Given a question and a context,
                answer the question step-by-step.

                If you are unsure, answer Unknown.
                """,
            },
            {
                "role": "user",
                "content": f"""
                    <question>
                    {query}
                    </question>
                    <context>
                    {formatted_context}
                    </context>
                    """,
            },
        ],
        response_model=ReasoningAndResponse,
    )


if __name__ == "__main__":
    query = "Can Arnold Schwarzenegger deadlift an adult Black rhinoceros?"
    decomposed_queries = asyncio.run(generate_queries(query))

    for generated_query in decomposed_queries.queries:
        print(generated_query)
        #> How much can Arnold Schwarzenegger deadlift?
        #> How much does an adult Black rhinoceros weigh?

    chains = asyncio.run(batch_reasoning_chains(decomposed_queries.queries))

    response = asyncio.run(generate_response(query, chains))

    print(response.model_dump_json(indent=2))
    """
    {
      "intermdiate_reasoning": "Arnold Schwarzenegger, during his prime
      bodybuilding years, was known for his impressive strength. However,
      the specific records of his deadlift are not well documented. The
      capacity to deadlift involves lifting a weight from the ground to
      the level of the hips before standing up straight with it. Even if
      Arnold's strength was substantial, lifting an adult Black rhinoceros,
      which weighs between 1,750 – 3,000 pounds (800 – 1,400 kg), would be
      an extraordinary feat. The heaviest documented deadlifts by top
      powerlifters are slightly above 1,000 pounds, far less than the weight
      of an adult Black rhinoceros, making it highly unlikely that Arnold
      Schwarzenegger could deadlift such an animal.",
      "correct_answer": "No"
    }
    """
```

### References

<sup id="ref-1">1</sup>: [Answering Questions by Meta-Reasoning over Multiple Chains of Thought](https://arxiv.org/pdf/2304.13007)
