import instructor
from enum import Enum
from pydantic import BaseModel, Field, field_validator
from openai import OpenAI

client = instructor.from_openai(OpenAI())


class TicketType(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    GENERAL = "general"


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TicketClassification(BaseModel):
    ticket_type: TicketType
    priority: Priority
    confidence: float = Field(..., description="Classification confidence")
    reason: str = Field(description="Brief reason for the classification")

    @field_validator("confidence")
    @classmethod
    def confidence_must_be_high(cls, v: float) -> float:
        if v < 0.5:
            raise ValueError(
                "Confidence must be at least 0.5, retry with a clearer classification"
            )
        return v


# Streaming: each chunk is a partially-filled TicketClassification
stream = client.chat.completions.create_partial(
    model="gpt-4.1-mini",
    max_retries=2,
    response_model=TicketClassification,
    messages=[
        {
            "role": "user",
            "content": "Classify this support ticket: 'I was charged twice for my subscription last month and need a refund immediately.'",
        }
    ],
)

for chunk in stream:
    print(chunk)
