import instructor

from instructor import openai_moderation

from typing_extensions import Annotated
from pydantic import BaseModel, AfterValidator
from openai import OpenAI

client = instructor.patch(OpenAI())


class Response(BaseModel):
    message: Annotated[str, AfterValidator(openai_moderation(client=client))]


response = Response(message="I want to make them suffer the consequences")
