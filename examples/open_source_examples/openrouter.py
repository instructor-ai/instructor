import os
import openai
from pydantic import BaseModel, Field
from typing import Optional
from instructor import Mode, patch, Maybe
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Extract API key from environment
openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY is not set in environment variables")

# Base URL for OpenAI
openrouter_base_url = "https://openrouter.ai/api/v1"

# Pydantic model for user details


class UserDetail(BaseModel):
    """Details in the text about the user"""
    name: str = Field(description="Name extracted from the text")
    age: int = Field(description="Age extracted from the text")
    occupation: Optional[str] = Field(
        default=None, description="Occupation extracted from the text")


def create_openai_client(api_key: str, base_url: str):
    """
    Create and patch OpenAI client.
    """
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    return patch(client, mode=Mode.JSON)


def get_user_details(client, model: str, user_message: dict) -> Maybe(UserDetail):
    """
    Extract user details using Open soure model.
    """
    maybe_user_detail = Maybe(UserDetail)
    system_message = {
        "role": "system",
        "content": f"You are an expert at outputting json. You always output valid pydantic model_json_schemas."
    }
    try:
        response = client.chat.completions.create(
            response_model=Maybe(UserDetail),
            response_format={"type": "json_object"},
            model=model,
            messages=[system_message, user_message]
        )
    except Exception as e:
        return maybe_user_detail(error=True, result=None, message=str(e))
    return response


def create_user_message(role: str, content: str) -> dict:
    maybe_detail = Maybe(UserDetail)
    return {
        "role": role,
        "content": f"Extract the user details from the following text: {content}. Match your response to this json_schema:{maybe_detail.model_json_schema()}"
    }


def create_retry_message(role: str, error: str, content: str) -> dict:
    return [{
        "role": role,
        "content": f"You experienced an error with the previous context: {error}. Please try again."
    }, {
        "role": "user",
        "content": f"Extract the user details from the following text: {content}"
    }]


# Initialize OpenAI client
client = create_openai_client(openrouter_api_key, openrouter_base_url)

# Define model and user message
model = "teknium/openhermes-2.5-mistral-7b"

# Create items to evaluate output
user_detail_messages = ["Brandon is 33 years old. He works as a solution architect.",
                        "Jason is 25 years old. He is the GOAT.", "Dominic is 45 years old. He is retired.", "Jenny is 72. She is a wife and a CEO.", "Holly is 22. She is an explorer.", "There onces was a prince, named Benny. He ruled for 10 years, which just ended. He started at 22.", "Simon says, why are you 22 years old marvin?"]


for message in user_detail_messages:
    user_message = create_user_message("user", message)
    user = get_user_details(client, model, user_message)
    # Assertions for validation
    if user.error:
        raise ValueError(user.error)
    if user.result:
        assert isinstance(user.result, UserDetail)
        print(user.result.name)
       
