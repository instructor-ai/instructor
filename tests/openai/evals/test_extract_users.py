import braintrust
import uuid
import pytest
from itertools import product
from pydantic import BaseModel
from openai import OpenAI
import instructor
from instructor.function_calls import Mode
import os


class UserDetails(BaseModel):
    name: str
    age: int


# Lists for models, test data, and modes
models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-1106-preview", "meta/llama-2-70b-chat"]
test_data = [
    ("Jason is 10", "Jason", 10),
    ("Alice is 25", "Alice", 25),
    ("Bob is 35", "Bob", 35),
]
modes = [Mode.FUNCTIONS, Mode.JSON, Mode.TOOLS]

run_id = uuid.uuid4().hex


@pytest.mark.parametrize("model, data, mode", product(models, test_data, modes))
def test_extract(model, data, mode):
    sample_data, expected_name, expected_age = data

    if mode == Mode.JSON and model in {"gpt-3.5-turbo", "gpt-4"}:
        pytest.skip(
            "JSON mode is not supported for gpt-3.5-turbo and gpt-4, skipping test"
        )
    if mode == Mode.FUNCTIONS and model == "meta/llama-2-70b-chat":
        pytest.skip(
            "FUNCTIONS mode is not supported for meta/llama-2-70b-chat, skipping test"
        )

    with braintrust.init(
        project="instructor_text_extract",
        experiment=f"{model}_{mode}_{run_id[:8]}",
        update=True,
        org_name="braintrustdata.com",
    ) as experiment:
        with experiment.start_span(
            input=sample_data, expected={"name": expected_name, "age": expected_age}
        ) as span:
            # Setting up the client with the instructor patch
            client = instructor.patch(
                braintrust.wrap_openai(
                    OpenAI(
                        base_url="https://proxy.braintrustapi.com/v1",
                        api_key=os.environ["BRAINTRUST_API_KEY"],
                    )
                ),
                mode=mode,
            )

            # Calling the extract function with the provided model, sample data, and mode
            response = client.chat.completions.create(
                model=model,
                response_model=UserDetails,
                messages=[
                    {"role": "user", "content": sample_data},
                ],
            )

            span.log(
                output=response.dict(),
                scores={
                    "name": response.name == expected_name,
                    "age": response.age == expected_age,
                },
            )

        print(experiment.summarize())
        # Assertions
        assert (
            response.name == expected_name
        ), f"Expected name {expected_name}, got {response.name}"
        assert (
            response.age == expected_age
        ), f"Expected age {expected_age}, got {response.age}"
