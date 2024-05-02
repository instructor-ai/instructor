from pydantic import BaseModel
from fastapi import FastAPI
from openai import AsyncOpenAI
import instructor
import logfire
from collections.abc import Iterable
from fastapi.responses import StreamingResponse


class UserData(BaseModel):
    query: str


class UserDetail(BaseModel):
    name: str
    age: int


app = FastAPI()
openai_client = AsyncOpenAI()
logfire.configure(pydantic_plugin=logfire.PydanticPlugin(record="failure"))
client = instructor.from_openai(openai_client)
logfire.instrument_fastapi(app)
client = instructor.from_openai(openai_client)


@app.post("/user", response_model=UserDetail)
async def endpoint_function(data: UserData) -> UserDetail:
    user_detail = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=UserDetail,
        messages=[
            {"role": "user", "content": f"Extract: `{data.query}`"},
        ],
    )
    logfire.info("/User returning", value=user_detail)
    return user_detail


@app.post("/extract", response_class=StreamingResponse)
async def extract(data: UserData):
    users = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=Iterable[UserDetail],
        stream=True,
        messages=[
            {"role": "user", "content": data.query},
        ],
    )

    async def generate():
        with logfire.span("Generating User Response Objects"):
            async for user in users:
                resp_json = user.model_dump_json()
                logfire.info("Returning user object", value=resp_json)

                yield resp_json

    return StreamingResponse(generate(), media_type="text/event-stream")
