import pytest
from pydantic import BaseModel
import instructor
from .util import modes, models

pdf_url = "https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/invoice.pdf"


class Invoice(BaseModel):
    total: float
    items: list[str]


@pytest.mark.parametrize("mode", modes)
@pytest.mark.parametrize("model", models)
def test_mistral_retry_validation(client, model, mode):
    client = instructor.from_mistral(client, mode=mode)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    "Extract information from the invoice.",
                    instructor.multimodal.PDF.from_url(pdf_url),
                ],
            }
        ],
        response_model=Invoice,
    )
    assert response.total == 220
    assert len(response.items) == 2


@pytest.mark.parametrize("mode", modes)
@pytest.mark.parametrize("model", models)
@pytest.mark.asyncio
async def test_mistral_retry_validation_async(client, model, mode):
    client = instructor.from_mistral(client, mode=mode, use_async=True)
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    "Extract information from the invoice.",
                    instructor.multimodal.PDF.from_url(pdf_url),
                ],
            }
        ],
        response_model=Invoice,
    )
    assert response.total == 220
    assert len(response.items) == 2
