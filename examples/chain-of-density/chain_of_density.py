from pydantic import BaseModel, Field, model_validator, FieldValidationInfo
from typing import List, Optional
import spacy
import instructor
import openai
from json import JSONDecodeError
from pydantic import ValidationError

instructor.patch()


class MissingEntity(BaseModel):
    """
    An entity is a real-world object that's assigned a name - for example, a person, country a product or a book title.

    A missing entity is:
    - relevant to the main story,
    - specific yet concise (5 words or fewer),
    - novel (not in the previous summary),
    - faithful (present in the article),
    - anywhere (can be located anywhere in the article).
    """

    entity_name: str = Field(
        ...,
        description="This is the associated name with the entity that exists in the text",
    )
    reason: str = Field(
        ...,
        description="This is a short sentence which describes why we should include this new entity in the rewritten abstract",
    )


class OmittedEntity(BaseModel):
    """
    An entity is a real-world object that's assigned a name - for example, a person, country a product or a book title.
    """

    entity_name: str = Field(
        ...,
        description="This is an entity which was present in the previous summary and not in the newly generated summary",
    )


class MaybeOmittedEntities(BaseModel):
    """
    This represents whether the new summary has omitted any entities that were present in the previous summary provided.
    """

    omitted_entities: Optional[List[OmittedEntity]] = Field(default=[])
    message: Optional[str] = Field(default=None)


MissingEntities = instructor.MultiTask(
    MissingEntity,
    name="missing-entities",
    description="""
    Identify 1-3 informative entities (";" delimited) from the article which are missing from the previously generated summary.
    """,
)


class InitialSummary(BaseModel):
    """
    This is an initial summary which should be long ( 4-5 sentences, ~80 words) yet highly non-specific, containing little information beyond the entities marked as missing. Use overly verbose languages and fillers (Eg. This article discusses) to reach ~80 words.
    """

    summary: str = Field(
        ...,
        description="This is a summary of the article provided which is overly verbose and has fillers to reach ~80 words",
    )


# Note that we utilise Spacy for entity recognition so that it is consistent with the original paper implementation which uses it as an original prompt
class RewrittenSummary(instructor.OpenAISchema):
    """
    This is a new,denser summary of identical length which covers every entity and detail from the previous summary plus the missing entities provided.

    Guidelines
    - Make every word count : Rewrite the previous summary to improve flow and make space for additional entities
    - Never remove an existing entity from the original text
    - Make space with fusion, compression, and removal of uninformative phrases like "the article discusses"
    - Missing entities can appear anywhere in the new summary
    """

    summary: str = Field(..., description="Concise yet self-contained summary")

    @model_validator(mode="after")
    def validate_sources(self, info: FieldValidationInfo) -> "RewrittenSummary":
        original_summary = info.context.get("original_summary")

        # We first extract out the original summary and compute the entity count
        evaluation: MaybeOmittedEntities = openai.ChatCompletion.create(
            model="gpt-4-0613",
            messages=[
                {
                    "role": "system",
                    "content": "You are about to be handed two summaries. Determine if the second summary has removed any entities that are present within the first summary",
                },
                {
                    "role": "user",
                    "content": f"Here is the first summary : {original_summary}",
                },
                {
                    "role": "user",
                    "content": f"Here is the second summary : {self.summary}",
                },
            ],
            response_model=MaybeOmittedEntities,
        )
        if evaluation.omitted_entities:
            omitted_entities = [
                entity.entity_name for entity in evaluation.omitted_entities
            ]
            raise ValueError(
                f"The following entities were omitted from the summary - {','.join(omitted_entities)}. Please generate a new summary which does not omit them."
            )

        return self


def rewrite_summary(
    article: str,
    existing_summary: str,
    entity_ctx: str,
    error_msgs: List[str] = [],
    remaining_retries=3,
):
    # # We then perform a new summary and validate that the entity density has increased ( We have not lost any entities )
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        functions=[RewrittenSummary.openai_schema],
        function_call={"name": RewrittenSummary.openai_schema["name"]},
        max_retries=2,
        messages=[
            {
                "role": "system",
                "content": "You are about to be given an article, an existing summary of the article and some new entities. Please use the information to rewrite the summary to make it denser and more concise so that it covers every entity and detail from the previous summary plus the missing entities",
            },
            {"role": "user", "content": f"Here is the article : {article}"},
            {
                "role": "user",
                "content": f"Here is the most recent article : {existing_summary}",
            },
            {
                "role": "user",
                "content": f"Here is some information on entities you should include in the rewritten summary: {entity_ctx}",
            },
            *error_msgs,
        ],
    )
    try:
        new_summary = RewrittenSummary.from_response(
            completion, validation_context={"prev_summary": existing_summary}
        )
        return new_summary
    except (ValidationError, JSONDecodeError) as e:
        if remaining_retries == 0:
            raise e
        error_msgs = []
        error_msgs.append(dict(**completion.choices[0].message))
        error_msgs.append(
            {
                "role": "user",
                "content": f"Recall the function correctly, exceptions found\n{e}",
            }
        )
        return rewrite_summary(
            article,
            existing_summary,
            entity_ctx,
            error_msgs,
            remaining_retries=remaining_retries - 1,
        )


def summarize_article(article: str, summary_steps: int = 3):
    summary_chain = []
    # We first generate an initial summary
    summary: InitialSummary = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        response_model=InitialSummary,
        messages=[
            {
                "role": "system",
                "content": "You will generate an increasingly concise, entity-dense summary of the following Article. ",
            },
            {"role": "user", "content": f"Here is the Article: {article}"},
        ],
        max_retries=2,
    )
    summary_chain.append(summary.summary)

    # # Then we perform a denser summarization
    for _ in range(summary_steps):
        # At the end of each step, we compare the most recent summarization and see what additional missing entities we need to perform
        missing_entities_response = openai.ChatCompletion.create(
            model="gpt-4-0613",
            stream=False,
            functions=[MissingEntities.openai_schema],
            function_call={"name": MissingEntities.openai_schema["name"]},
            messages=[
                {"role": "user", "content": f"Here is the Article: {article}"},
                {
                    "role": "user",
                    "content": f"Here is the summary: {summary_chain[-1]}",
                },
            ],
            max_retries=2,
            max_tokens=1000,
        )
        missing_entities: List[MissingEntities] = [
            i for i in MissingEntities.from_response(missing_entities_response)
        ][0][1]

        entity_ctx = ""
        for entity in missing_entities:
            assert isinstance(entity, MissingEntity)
            entity_ctx += f"\n-{entity.entity_name} : {entity.reason}"
        new_summary = rewrite_summary(article, summary_chain[-1], entity_ctx)
        summary_chain.append(new_summary.summary)

    return summary_chain
