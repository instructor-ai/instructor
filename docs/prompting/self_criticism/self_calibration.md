---
description: "Self-Refine involves getting LLMs to iteratively generate new responses based off feedback or unitl a stopping condition is met"
---

Self Refine <sup><a href="https://arxiv.org/pdf/2303.17651">1</a></sup> involves prompting a LLM to provide feedback on an answer. This iterative process continues until a stopping condition is met.

We can implement this using `instructor` as seen below using our validation context below.

```python hl_lines="25-27 56-65"
import instructor
from openai import OpenAI
from pydantic import BaseModel, field_validator, ValidationInfo
from typing import Literal

client = instructor.from_openai(OpenAI())


class Sentiment(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str, info: ValidationInfo) -> str:
        if not v or not v.strip():
            raise ValueError("Text must not be empty or whitespace")
        pairwise_comparison_result = validate_sentiment(
            v,
            info.context["reference_statement"],  # type: ignore
            info.context["sentiment"],  # type: ignore
        )

        if pairwise_comparison_result.alignment_result == "Review B":
            raise ValueError(
                f"""{pairwise_comparison_result.feedback}. Please modify
                your statement to be more aligned with the target sentiment
                and do not copy the statement provided for reference"""
            )

        if v == info.context["reference_statement"]:  # type: ignore
            raise ValueError(
                """Your statement is the same as the reference statement.
                It should be a separate statement from the reference
                statement."""
            )

        return v


class PairwiseEvaluation(BaseModel):
    feedback: str
    alignment_result: Literal[
        "Review A",
        "Review B",
        "Both",
    ]


def validate_sentiment(review_a: str, review_b: str, target_sentiment: str):
    return client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": f"""
                Which review is aligned with the sentiment
                {target_sentiment}?

                Review A: {review_a}
                Review B: {review_b}.

                Pick your answer from ['Review A', 'Review B', 'both',
                'neither']. Generate a short explanation for your choice
                first. Then generate your response on which review is more
                aligned
                """,
            }
        ],
        response_model=PairwiseEvaluation,
    )


def generate_sentiment_analysis(
    initial_statement: str, target_sentiment: str, reference_statement: str
) -> Sentiment:
    return client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """
                You are an expert at sentiment analysis.
                Rewrite a statement so that it is more
                closely aligned with the target sentiment.
                """,
            },
            {
                "role": "user",
                "content": f"""
                The statement is {initial_statement} and
                the desired target sentiment is
                {target_sentiment}
                """,
            },
        ],
        response_model=Sentiment,
        validation_context={
            "sentiment": target_sentiment,
            "reference_statement": reference_statement,
        },
        max_retries=5,
    )


if __name__ == "__main__":
    aligned_sentiment = generate_sentiment_analysis(
        """The food was fantastic, with every dish
        surpassing our expectations in terms of flavor,
        presentation, and overall dining experience.""",
        "Negative",
        """The food was awful, with each dish failing to
        meet even the most basic standards of taste,
        quality, and presentation, resulting in a highly
        disappointing dining experience.""",
    )
    print(aligned_sentiment)
    """
    text = 'The food was terrible, with every dish failing to meet our
    expectations in terms of flavor, presentation, and overall dining
    experience.'
    """
```

### References

<sup id="ref-1">1</sup>: [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/pdf/2303.17651)
