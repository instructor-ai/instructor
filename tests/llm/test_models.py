"""Common test models used across all provider test suites.

This module contains Pydantic models that are commonly used in tests,
reducing duplication across provider-specific test files.
"""

from typing import Optional, Literal
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.functional_validators import AfterValidator
from typing import Annotated


# Basic extraction models
class UserExtract(BaseModel):
    """Basic user extraction model used in many tests."""

    name: str
    age: int


class User(BaseModel):
    """Simple user model."""

    name: str
    age: int


# Validation models
def uppercase_validator(v: str) -> str:
    """Validator that checks if string is uppercase."""
    if not v.isupper():
        raise ValueError("Name must be uppercase")
    return v.strip()


class UserDetail(BaseModel):
    """User model with validation."""

    name: Annotated[str, AfterValidator(uppercase_validator)]
    age: int


class UserWithValidator(BaseModel):
    """User model with field validator."""

    name: str
    age: int

    @field_validator("name")
    def name_must_be_uppercase(cls, v: str) -> str:
        if not v.isupper():
            raise ValueError("Name must be uppercase")
        return v


# Nested models
class Item(BaseModel):
    """Item in an order."""

    name: str
    price: float
    quantity: int = 1


class Order(BaseModel):
    """Order with nested items."""

    order_id: str
    customer_name: str
    items: list[Item]
    total: Optional[float] = None

    @model_validator(mode="after")
    def calculate_total(self):
        if self.items:
            self.total = sum(item.price * item.quantity for item in self.items)
        return self


# Classification models
class Labels(str, Enum):
    """Binary classification labels."""

    SPAM = "spam"
    NOT_SPAM = "not_spam"


class MultiLabel(str, Enum):
    """Multi-class classification labels."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class SinglePrediction(BaseModel):
    """Single classification prediction."""

    class_label: Labels


class MultiClassPrediction(BaseModel):
    """Multi-class prediction with confidence."""

    class_label: MultiLabel
    confidence: float = Field(ge=0.0, le=1.0)


# Streaming models
class PartialUser(BaseModel):
    """User model that supports partial/streaming responses."""

    name: Optional[str] = None
    age: Optional[int] = None
    bio: Optional[str] = None


# Complex extraction models
class Address(BaseModel):
    """Address component."""

    street: str
    city: str
    state: str
    zip_code: str


class Company(BaseModel):
    """Company information."""

    name: str
    industry: str
    employee_count: Optional[int] = None


class Contact(BaseModel):
    """Complex contact information."""

    name: str
    email: str
    phone: Optional[str] = None
    address: Optional[Address] = None
    company: Optional[Company] = None


# Response models for specific test scenarios
class MovieReview(BaseModel):
    """Movie review for sentiment analysis."""

    title: str
    rating: int = Field(ge=1, le=5)
    review_text: str
    sentiment: Literal["positive", "negative", "neutral"]


class Entity(BaseModel):
    """Named entity for NER tests."""

    text: str
    label: Literal["PERSON", "ORGANIZATION", "LOCATION", "DATE"]
    start_pos: int
    end_pos: int


class ExtractedEntities(BaseModel):
    """Container for extracted entities."""

    entities: list[Entity]


# Test data associated with models
CLASSIFICATION_TEST_DATA = [
    ("I am a spammer", Labels.SPAM),
    ("I am not a spammer", Labels.NOT_SPAM),
    ("Buy cheap products now!", Labels.SPAM),
    ("Hello, how are you?", Labels.NOT_SPAM),
]

MULTI_CLASS_TEST_DATA = [
    ("I love this product!", MultiLabel.POSITIVE),
    ("This is terrible", MultiLabel.NEGATIVE),
    ("It's okay, nothing special", MultiLabel.NEUTRAL),
]

USER_EXTRACTION_PROMPTS = [
    "Extract: Jason is 25 years old",
    "Extract: Sarah is 30 years old",
    "Extract: Mike is 28 years old",
]

NESTED_EXTRACTION_PROMPT = """
Extract this order:
Order ID: ORD-001
Customer: John Doe
Items:
- Widget A: $10.00 x 2
- Gadget B: $25.00 x 1
"""
