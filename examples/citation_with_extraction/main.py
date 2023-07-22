from typing import Iterable, List, Optional
from fastapi import FastAPI
from openai_function_call import MultiTask
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

import openai


# FastAPI app
app = FastAPI(
    title="Citation with Extraction",
)


class SubResponse(BaseModel):
    """
    If there are multiple phrases with difference citations. Each one should be its own object.
    make sure to break them apart such that each one only uses a set of
    sources that are relevant to it. If you need to say something without a citation leave the quotes as None
    """

    body: str = Field(..., description="Body of the sentences, as part of a response")
    substring_quotes: Optional[List[str]] = Field(
        ...,
        description="Each source should be a direct quote from the context, as a substring of the original content but should be a wide enough quote to capture the context of the quote. The citation should at least be long and capture the context and be a full sentence.",
    )

    def _get_span(self, quote, context, errs=100):
        import regex

        minor = quote
        major = context

        errs_ = 0
        s = regex.search(f"({minor}){{e<={errs_}}}", major)
        while s is None and errs_ <= errs:
            errs_ += 1
            s = regex.search(f"({minor}){{e<={errs_}}}", major)

        if s is not None:
            yield from s.spans()

    def get_spans(self, context):
        if self.substring_quotes:
            for quote in self.substring_quotes:
                yield from self._get_span(quote, context)


Answers = MultiTask(
    SubResponse,
    name="Answer",
    description="Correctly answer questions based on a context. Quotes should be full sentences when possible",
)


class Question(BaseModel):
    context: str = Field(..., description="Context to extract answers from")
    query: str = Field(..., description="Question to answer")


# Function to extract entities from input text using GPT-3.5
def stream_extract(question: Question) -> Iterable[SubResponse]:
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        temperature=0,
        stream=True,
        functions=[Answers.openai_schema],
        function_call={"name": Answers.openai_schema["name"]},
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
    return Answers.from_streaming_response(completion)


# Route to handle SSE events and return users
@app.post("/extract", response_class=StreamingResponse)
async def extract(question: Question):
    facts = stream_extract(question)

    async def generate():
        for fact in facts:
            spans = list(fact.get_spans(question.context))
            resp = {
                "body": fact.body,
                "spans": spans,
                "citation": fact.substring_quotes,
            }
            yield f"data: {resp}"

    return StreamingResponse(generate(), media_type="text/event-stream")
