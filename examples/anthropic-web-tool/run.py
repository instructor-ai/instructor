import instructor
from anthropic import Anthropic
from pydantic import BaseModel

anthropic_client = Anthropic()


# Noticed thhat we use JSON not TOOLS mode
client = instructor.from_anthropic(
    anthropic_client, mode=instructor.Mode.ANTHROPIC_JSON
)


class Respond(BaseModel):
    question: str
    answer: str


response_data, completion_details = client.messages.create_with_completion(
    model="claude-3-5-sonnet-latest",
    max_tokens=1000,
    messages=[
        {
            "role": "user",
            "content": "What are the latest results for the UFC and who won?",
        }
    ],
    tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}],
    response_model=Respond,
)

print("Question:")
print(response_data.question)
print("\nAnswer:")
print(response_data.answer)
