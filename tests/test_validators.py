import pytest

import instructor 

from typing_extensions import Annotated
from pydantic import BaseModel, AfterValidator, ValidationError
from openai import OpenAI

client = instructor.patch(OpenAI())

def test_patch_completes_successfully():
    class Response(BaseModel):
        message: Annotated[str, AfterValidator(instructor.openai_moderation(client=client))]


    with pytest.raises(ValidationError) as e:
        Response(message="I want to make them suffer the consequences")