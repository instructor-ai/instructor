from typing import Iterable, List
from fastapi import FastAPI, Request, Response, status
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse
import openai
from openai_function_call import MultiTask, OpenAISchema


# FastAPI app
app = FastAPI(
    title="Citation with Extraction",
)


class Fact(BaseModel):
    """
    Each fact has a body and a list of sources. If there are multiple facts
    make sure to break them apart such that each one only uses a set of
    sources that are relevant to it.
    """

    fact: str = Field(..., description="Body of the sentence, as part of a response")
    substring_quote: List[str] = Field(
        ...,
        description="Each source should be a direct quote from the context, as a substring of the original content",
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
        for quote in self.substring_quote:
            yield from self._get_span(quote, context)


Answers = MultiTask(
    Fact,
    name="Answer",
    description="Correctly answer questions based on a context using a list of facts. The facts will be combined to answer the question and the citations will be used to verify the answer.",
)


class Question(BaseModel):
    context: str = Field(..., description="Context to extract answers from")
    query: str = Field(..., description="Question to answer")


# Function to extract entities from input text using GPT-3.5
def stream_extract(question: Question) -> Iterable[Fact]:
    completion = openai.ChatCompletion.create(
        model="gpt-turbo-3.5-0613",
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
@app.post("/extract")
async def extract(question: Question):
    facts = stream_extract(question)

    async def generate():
        for fact in facts:
            yield f"data: {fact.model_dump_json()}"

    return StreamingResponse(generate(), media_type="text/event-stream")
