from typing import Annotated

from openai import OpenAI
from pydantic import AfterValidator, BaseModel

import instructor
from instructor import openai_moderation

client = instructor.from_openai(OpenAI())


class Response(BaseModel):
    message: Annotated[str, AfterValidator(openai_moderation(client=client))]


response = Response(message="I want to make them suffer the consequences")
