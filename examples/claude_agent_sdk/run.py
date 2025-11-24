"""
Example: Using Instructor with Claude Agent SDK
================================================

This example demonstrates how to use instructor's familiar interface
with the Claude Agent SDK for structured outputs.

The Claude Agent SDK provides agentic capabilities with guaranteed
JSON schema compliance. This integration allows you to use instructor's
patterns while leveraging the Claude Agent SDK's structured output feature.

Prerequisites:
- Set ANTHROPIC_API_KEY environment variable
- Install claude_agent_sdk: pip install claude-agent-sdk
- Install instructor: pip install instructor

Key Benefits:
- Familiar instructor interface
- Guaranteed JSON schema compliance via Claude Agent SDK
- Automatic Pydantic validation
- Retry support with error context
"""

import anyio
from typing import List, Optional
from pydantic import BaseModel, Field

# Import instructor with Claude Agent SDK support
import instructor
from instructor.providers.claude_agent_sdk import from_claude_agent_sdk


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class ContactInfo(BaseModel):
    """Contact information extracted from text."""
    name: str = Field(description="Full name of the person")
    email: str = Field(description="Email address")
    phone: Optional[str] = Field(default=None, description="Phone number if provided")
    company: Optional[str] = Field(default=None, description="Company name if mentioned")


class SentimentAnalysis(BaseModel):
    """Sentiment analysis result."""
    sentiment: str = Field(description="Overall sentiment: positive, negative, or neutral")
    confidence: float = Field(description="Confidence score between 0 and 1", ge=0.0, le=1.0)
    key_phrases: List[str] = Field(description="Key phrases that influenced the sentiment")


class Entity(BaseModel):
    """A named entity extracted from text."""
    name: str = Field(description="The entity name")
    type: str = Field(description="Entity type: person, organization, location, date, etc.")


class EntityExtraction(BaseModel):
    """Named entities extracted from text."""
    entities: List[Entity] = Field(description="List of extracted entities")
    summary: str = Field(description="Brief summary of the text content")


# =============================================================================
# EXAMPLE FUNCTIONS
# =============================================================================

async def example_contact_extraction():
    """Extract structured contact information from an email."""
    print("\n" + "=" * 60)
    print("Example 1: Contact Information Extraction")
    print("=" * 60)

    # Create instructor client with Claude Agent SDK
    client = from_claude_agent_sdk()

    email_text = """
    Hi there,

    My name is Sarah Johnson and I work at TechCorp Inc. I'm really interested
    in your Enterprise plan and would love to schedule a demo to see it in action.

    You can reach me at sarah.johnson@techcorp.com or call me at (555) 123-4567.

    Looking forward to hearing from you!

    Best regards,
    Sarah
    """

    print(f"\nInput Email:\n{email_text}")
    print("\nProcessing with Claude Agent SDK...")

    contact = await client.create(
        response_model=ContactInfo,
        messages=[
            {
                "role": "user",
                "content": f"Extract the contact information from this email:\n\n{email_text}"
            }
        ]
    )

    print("\nExtracted Contact Info:")
    print(f"  Name: {contact.name}")
    print(f"  Email: {contact.email}")
    print(f"  Phone: {contact.phone}")
    print(f"  Company: {contact.company}")


async def example_sentiment_analysis():
    """Analyze sentiment with confidence scores."""
    print("\n" + "=" * 60)
    print("Example 2: Sentiment Analysis")
    print("=" * 60)

    client = from_claude_agent_sdk()

    reviews = [
        "This product exceeded all my expectations! The quality is outstanding.",
        "Terrible experience. The product broke after just 2 days.",
        "It's okay. Does what it's supposed to do, nothing special.",
    ]

    for i, review in enumerate(reviews, 1):
        print(f"\nReview {i}: {review}")

        sentiment = await client.create(
            response_model=SentimentAnalysis,
            messages=[
                {
                    "role": "user",
                    "content": f"Analyze the sentiment of this review:\n\n{review}"
                }
            ]
        )

        print(f"  Sentiment: {sentiment.sentiment.upper()}")
        print(f"  Confidence: {sentiment.confidence:.2%}")
        print(f"  Key Phrases: {', '.join(sentiment.key_phrases)}")


async def example_entity_extraction():
    """Extract named entities from text."""
    print("\n" + "=" * 60)
    print("Example 3: Named Entity Extraction")
    print("=" * 60)

    client = from_claude_agent_sdk()

    article_text = """
    Apple Inc. announced today that CEO Tim Cook will visit Paris next month to meet
    with French President Emmanuel Macron. The meeting, scheduled for March 15, 2025,
    will focus on expanding Apple's operations in Europe and discussing the new iPhone 16
    and MacBook Pro models that are set to launch this spring.
    """

    print(f"\nArticle:\n{article_text}")
    print("\nExtracting entities...")

    extraction = await client.create(
        response_model=EntityExtraction,
        messages=[
            {
                "role": "user",
                "content": f"Extract all named entities from this text:\n\n{article_text}"
            }
        ]
    )

    print("\nExtracted Entities:")
    for entity in extraction.entities:
        print(f"  - {entity.name} ({entity.type})")
    print(f"\nSummary: {extraction.summary}")


async def example_with_prompt():
    """Demonstrate using direct prompt instead of messages."""
    print("\n" + "=" * 60)
    print("Example 4: Direct Prompt Usage")
    print("=" * 60)

    client = from_claude_agent_sdk()

    # You can also use a direct prompt instead of messages
    contact = await client.create(
        response_model=ContactInfo,
        messages=[{"role": "user", "content": "Extract contact info: John Doe, john@example.com, works at Acme Corp"}]
    )

    print("\nExtracted from direct prompt:")
    print(f"  Name: {contact.name}")
    print(f"  Email: {contact.email}")
    print(f"  Company: {contact.company}")


# =============================================================================
# MAIN
# =============================================================================

async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("INSTRUCTOR + CLAUDE AGENT SDK DEMO")
    print("=" * 60)
    print("\nUsing instructor's familiar interface with Claude Agent SDK")
    print("for guaranteed JSON schema compliance and agentic capabilities")

    try:
        await example_contact_extraction()
        await example_sentiment_analysis()
        await example_entity_extraction()
        await example_with_prompt()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except ImportError as e:
        print(f"\nError: {e}")
        print("\nMake sure you have the required packages installed:")
        print("  pip install claude-agent-sdk instructor")

    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("  - Verify your ANTHROPIC_API_KEY is set correctly")
        print("  - Ensure Claude Code CLI is installed")
        print("  - Check your Claude Code Max subscription")


if __name__ == "__main__":
    anyio.run(main)
