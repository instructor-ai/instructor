import os
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional
from instructor import Maybe, Mode
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Extract API key from environment
openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY is not set in environment variables")

# Base URL for OpenAI
openrouter_base_url = os.environ.get("OPENROUTER_BASE_URL")
if not openrouter_base_url:
    raise ValueError("OPENROUTER_BASE_URL is not set in environment variables")

# Pydantic model for user details


class UserDetail(BaseModel):
    """Details in the text about the user"""

    name: str = Field(description="Name extracted from the text")
    age: int = Field(description="Age extracted from the text")
    occupation: Optional[str] = Field(
        default=None, description="Occupation extracted from the text"
    )


# Initialize OpenAI client
client = instructor.patch(
    OpenAI(api_key=openrouter_api_key, base_url=openrouter_base_url),
    mode=Mode.JSON,
)

# Wrap Userdetail in Maybe BaseModel
MaybeUser = Maybe(UserDetail)


def get_user_details(model: str, user_message: dict) -> MaybeUser:
    """
    Extract user details using Open soure model.
    """

    system_message = {
        "role": "system",
        "content": f"You are an expert at outputting json. You always output valid pydantic model_json_schemas.",
    }

    response = client.chat.completions.create(
        response_model=MaybeUser, model=model, messages=[system_message, user_message]
    )

    return response


def create_user_message(content: str, error=False) -> dict:
    if error:
        return [
            {
                "role": "system",
                "content": f"You experienced an error with the previous context: {error}. Please try again.",
            },
            {
                "role": "user",
                "content": f"Extract the user details from the following text: {content}",
            },
        ]
    return {
        "role": "user",
        "content": f"Extract the user details from the following text: {content}. Match your response to this json_schema:{MaybeUser.model_json_schema()}",
    }


# Define model and user message
model = "teknium/openhermes-2.5-mistral-7b"

# Create items to evaluate output
user_detail_messages = [
    "Brandon is 33 years old. He works as a solution architect.",
    "Jason is 25 years old. He is the GOAT.",
    "Dominic is 45 years old. He is retired.",
    "Jenny is 72. She is a wife and a CEO.",
    "Holly is 22. She is an explorer.",
    "There onces was a prince, named Benny. He ruled for 10 years, which just ended. He started at 22.",
    "Simon says, why are you 22 years old marvin?",
]


for message in user_detail_messages:
    user_message = create_user_message(message)
    user = get_user_details(model, user_message)
    # Assertions for validation
    if user.error:
        retry_message = create_user_message(message, user.message)
        user = get_user_details(model, retry_message)
        if user.result:
            print(user.result)
        else:
            raise ValueError(user.message)

    if user.result:
        print(user.result)
