import instructor
import openai
from pydantic import BaseModel, Field

from pprint import pprint
from pydantic import BaseModel, Field
from typing import List, Dict


class Summary(BaseModel):
    """Represents a summary entry in the list.

    Guidelines:
        - The first summary should be long (4-5 sentences, ~80 words) yet highly non-specific,
          containing little information beyond the entities marked as missing. Use overly verbose
          language and fillers (e.g., "this article discusses") to reach ~80 words.
        - Make every word count: rewrite the previous summary to improve flow and make space for
          additional entities.
        - Make space with fusion, compression, and removal of uninformative phrases like "the article discusses."
        - The summaries should become highly dense and concise yet self-contained, i.e., easily understood
          without the article.
        - Missing entities can appear anywhere in the new summary.
        - Never drop entities from the previous summary. If space cannot be made, add fewer new entities.
    """

    index: int = Field(..., description="Index of the summary in the chain.")
    denser_summary: str = Field(..., description="Concise yet self-contained summary.")
    included_entities: List[str] = Field(
        ..., description="Correct list of Entities found in the summary."
    )
    missing_entities: List[str] = Field(
        ...,
        description="Correct list of Entities found absent from the summary that should be included in the next summary attempt.",
    )


# This multitask helper will be used to generate a chain of summaries.
# Allows us to extract data via streaming to see resuls faster
def summarize_article(article: str, n_summaries: int = 5, stream: bool = True):
    ChainOfDenseSummaries = instructor.MultiTask(
        Summary,
        name="chain-of-dense-summaries",
        description=f"""
            Repeat the following 2 steps {n_summaries} times.

                Step 1. Identify 1-3 informative entities (";" delimited) from the article which are missing from the previously generated summary.

                Step 2. Write a new, denser summary of identical length which covers every entity and detail from the previous summary plus the missing entities.

                A missing entity is:

                - relevant to the main story,
                - specific yet concise (5 words or fewer),
                - novel (not in the previous summary),
                - faithful (present in the article),
                - anywhere (can be located anywhere in the article).

                Remember, use the exact same number of words for each summary.""",
    )
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        stream=stream,
        messages=[
            {
                "role": "system",
                "content": """Summarize the following article with {n_summaries} chain of summaries with increasing density:""",
            },
            {"role": "user", "content": article},
        ],
        functions=[ChainOfDenseSummaries.openai_schema],
        function_call={"name": ChainOfDenseSummaries.openai_schema["name"]},
    )
    if stream:
        return ChainOfDenseSummaries.from_streaming_response(completion)
    return ChainOfDenseSummaries.from_response(completion)
