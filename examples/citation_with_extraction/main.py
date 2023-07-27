import json
from typing import Iterable, List
from fastapi import FastAPI, Request, HTTPException
from fastapi.params import Depends
from openai_function_call import MultiTask, OpenAISchema
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

import os
import openai
import logging

from openai_function_call.dsl.multitask import MultiTaskBase

logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Citation with Extraction",
)


class Fact(BaseModel):
    """
    Class representing single statement.
    Each fact has a body and a list of sources.
    If there are multiple facts make sure to break them apart such that each one only uses a set of sources that are relevant to it.
    """

    fact: str = Field(
        ...,
        description="Body of the sentences, as part of a response, it should read like a sentence that answers the question",
    )
    substring_quotes: List[str] = Field(
        ...,
        description="Each source should be a direct quote from the context, as a substring of the original content",
    )

    def _get_span(self, quote, context):
        import regex

        minor = quote
        major = context

        errs_ = 0
        s = regex.search(f"({minor}){{e<={errs_}}}", major)
        while s is None and errs_ <= len(context) * 0.05:
            errs_ += 1
            s = regex.search(f"({minor}){{e<={errs_}}}", major)

        if s is not None:
            yield from s.spans()

    def get_spans(self, context):
        if self.substring_quotes:
            for quote in self.substring_quotes:
                yield from self._get_span(quote, context)


class QuestionAnswer(OpenAISchema, MultiTaskBase):
    """
    Class representing a question and its answer as a list of facts each one should have a soruce.
    each sentence contains a body and a list of sources."""

    question: str = Field(..., description="Question that was asked")
    tasks: List[Fact] = Field(
        ...,
        description="Body of the answer, each fact should be its seperate object with a body and a list of sources",
    )


QuestionAnswer.task_type = Fact


class Question(BaseModel):
    context: str = Field(..., description="Context to extract answers from")
    query: str = Field(..., description="Question to answer")


# Function to extract entities from input text using GPT-3.5
def stream_extract(question: Question) -> Iterable[Fact]:
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        temperature=0,
        stream=True,
        functions=[QuestionAnswer.openai_schema],
        function_call={"name": QuestionAnswer.openai_schema["name"]},
        messages=[
            {
                "role": "system",
                "content": f"You are a world class algorithm to answer questions with correct and exact citations. ",
            },
            {"role": "user", "content": f"Answer question using the following context"},
            {"role": "user", "content": f"{question.context}"},
            {"role": "user", "content": f"Question: {question.query}"},
            {
                "role": "user",
                "content": f"Tips: Make sure to cite your sources, and use the exact words from the context.",
            },
        ],
        max_tokens=2000,
    )
    return QuestionAnswer.from_streaming_response(completion)


def get_api_key(request: Request):
    """
    This just gets the API key from the request headers.
    but tries to read from the environment variable OPENAI_API_KEY first.
    """
    if "OPENAI_API_KEY" in os.environ:
        return os.environ["OPENAI_API_KEY"]

    auth = request.headers.get("Authorization")
    if auth is None:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    if auth.startswith("Bearer "):
        return auth.replace("Bearer ", "")

    return None


# Route to handle SSE events and return users
@app.post("/extract", response_class=StreamingResponse)
async def extract(question: Question, openai_key=Depends(get_api_key)):
    openai.api_key = openai_key
    facts = stream_extract(question)

    async def generate():
        for fact in facts:
            logger.info(f"Fact: {fact}")
            spans = list(fact.get_spans(question.context))
            resp = {
                "body": fact.fact,
                "spans": spans,
                "citation": [question.context[a:b] for (a, b) in spans],
            }
            resp_json = json.dumps(resp)
            yield f"data: {resp_json}"
        yield "data: [DONE]"

    return StreamingResponse(generate(), media_type="text/event-stream")
