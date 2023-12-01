#

import os
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional
from instructor import Mode

# Extract API key from environment
runpod_api_key = os.environ.get("RUNPOD_API_KEY")
assert runpod_api_key, "RUNPOD_API_KEY is not set in environment variables"

# Base URL for OpenAI
runpod_base_url = os.environ.get("RUNPOD_BASE_URL")
assert runpod_base_url, "RUNPOD_BASE_URL is not set in environment variables"

# Initialize OpenAI client
client = instructor.patch(
    OpenAI(api_key=runpod_api_key, base_url=runpod_base_url),
    mode=Mode.JSON,
)

# For direct reference here. See https://docs.perplexity.ai/docs/model-cards for updates
# Recommended is pplx-70b-chat
models = [
    "TheBloke_OpenHermes-2.5-Mistral-7B-GPTQ",
]

data = [
    "Brandon is 33 years old. He works as a solution architect.",
    "Jason is 25 years old. He is the GOAT.",
    "Dominic is 45 years old. He is retired.",
    "Jenny is 72. She is a wife and a CEO.",
    "Holly is 22. She is an explorer.",
    "There onces was a prince, named Benny. He ruled for 10 years, which just ended. He started at 22.",
    "Simon says, why are you 22 years old marvin?",
]


if __name__ == "__main__":

    class UserDetail(BaseModel):
        name: str = Field(description="Name extracted from the text")
        age: int = Field(description="Age extracted from the text")
        occupation: Optional[str] = Field(
            default=None, description="Occupation extracted from the text"
        )

    for content in data:
        prompt_template = f"""
            <|im_start|>system
            You are an expert at outputting json. You output valid JSON.<|im_end|>
            <|im_start|>user
            Extract the user details from the following text: {content}. Match your response to the following schema: {UserDetail.model_json_schema()}<|im_end|>
            <|im_start|>assistant
            """
        try:
            user = client.chat.completions.create(
                response_model=UserDetail,
                model="TheBloke_OpenHermes-2.5-Mistral-7B-GPTQ",
                messages=[
                    {
                        "role": "user",
                        "content": prompt_template,
                    }
                ],
                max_retries=1,
            )
            print(f"Result: {user}")
        except Exception as e:
            print(f"Error: {e}")
            continue
