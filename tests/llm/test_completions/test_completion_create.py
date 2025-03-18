import pytest
import instructor
from pydantic import BaseModel
from cohere import Client
from cerebras.cloud.sdk import Cerebras
from openai import OpenAI
from anthropic import Anthropic
from mistralai import Mistral
from google.generativeai import GenerativeModel as GenerativeAIGenerativeModel
import os
from vertexai.generative_models import GenerativeModel
from writerai import Writer
from fireworks.client import Fireworks
from google.genai import Client as GoogleGenaiClient


class User(BaseModel):
    name: str
    age: int


response_model_and_prompt = [
    (User, "Ivan is 20 and lives in Singapore?"),
    (bool, "Is 30 greater than 20?"),
]
clients = [
    {
        "client": instructor.from_openai(OpenAI()),
        "model": "gpt-4o-mini",
    },
    {
        "client": instructor.from_anthropic(Anthropic()),
        "model": "claude-3-5-haiku-20241022",
        "max_tokens": 2048,
    },
    {
        "client": instructor.from_gemini(
            GenerativeAIGenerativeModel(model_name="gemini-2.0-flash")
        ),
    },
    {
        "client": instructor.from_vertexai(
            GenerativeModel(model_name="gemini-2.0-flash")
        ),
    },
    {
        "client": instructor.from_mistral(
            Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        ),
        "model": "ministral-3b-latest",
    },
    {
        "client": instructor.from_writer(Writer(api_key=os.getenv("WRITER_API_KEY"))),
        "model": "palmyra-x-004",
    },
    {
        "client": instructor.from_fireworks(
            Fireworks(api_key=os.getenv("FIREWORKS_API_KEY"))
        ),
        "model": "accounts/fireworks/models/llama-v3-70b-instruct",
    },
    {
        "client": instructor.from_cohere(Client()),
        "model": "command-r-plus",
    },
    {
        "client": instructor.from_cerebras(Cerebras()),
        "model": "llama-3.3-70b",
    },
    {
        "client": instructor.from_genai(GoogleGenaiClient()),
        "model": "gemini-2.0-flash",
    },
]


@pytest.mark.parametrize(
    "response_model_and_prompt_item, kwargs",
    [
        pytest.param(
            item,
            client,
            id=f"{item[0].__name__}-{client.get('client').provider}",
        )
        for item in response_model_and_prompt
        for client in clients
    ],
)
def test_completions_sync(response_model_and_prompt_item, kwargs):  #
    response_model, prompt = response_model_and_prompt_item

    if kwargs.get("model", None) == "palmyra-x-004" and response_model == bool:
        pytest.skip("Palmyra-X-004 does not support boolean response models")
        return

    response, completion = kwargs["client"].chat.completions.create_with_completion(
        **{k: v for k, v in kwargs.items() if k != "client"},
        messages=[{"role": "user", "content": prompt}],
        response_model=response_model,
    )
    assert isinstance(response, response_model)


@pytest.mark.parametrize("kwargs", clients)
def test_completions_sync_list(kwargs):
    response, completion = kwargs["client"].chat.completions.create_with_completion(
        response_model=list[User],
        messages=[
            {
                "role": "user",
                "content": "Ivan is 20 and lives in Singapore, Jack is 30 and lives in New York while Jane is 25 and lives in London",
            }
        ],
        **{k: v for k, v in kwargs.items() if k != "client"},
    )
    for item in response:
        assert isinstance(item, User)
