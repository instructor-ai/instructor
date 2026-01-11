"""TOON Mode Examples

TOON (Token-Oriented Object Notation) is a compact format that achieves
30-60% token reduction compared to JSON while maintaining structured outputs.

Usage:
    pip install 'instructor[toon]'
    python run.py
"""

import time
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class User(BaseModel):
    name: str
    age: int
    email: str


class Address(BaseModel):
    street: str
    city: str
    zip: str


class UserWithAddress(BaseModel):
    user: User
    address: Address


class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Task(BaseModel):
    title: str = Field(description="Task title")
    description: str = Field(description="Task description")
    priority: Priority = Field(description="Task priority level")
    estimated_hours: float = Field(description="Estimated hours to complete")
    tags: list[str] = Field(description="List of tags")


class Project(BaseModel):
    name: str = Field(description="Project name")
    status: Literal["active", "completed", "on_hold"] = Field(
        description="Project status"
    )
    tasks: list[Task] = Field(description="List of project tasks")


def run_openai_example():
    """Run TOON mode with OpenAI."""
    from openai import OpenAI
    import instructor
    from instructor import Mode

    print("\n" + "=" * 60)
    print("OpenAI TOON Mode")
    print("=" * 60)

    client = instructor.from_openai(OpenAI(), mode=Mode.TOON)

    # Simple extraction
    print("\n1. Simple extraction:")
    user = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": "Extract: John is 30 and his email is john@example.com",
            }
        ],
        response_model=User,
    )
    print(f"   User: {user.name}, {user.age}, {user.email}")

    # Nested model
    print("\n2. Nested model:")
    result = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": "Extract: Alice lives at 123 Main St, New York, 10001. She is alice@test.com and 25 years old.",
            }
        ],
        response_model=UserWithAddress,
    )
    print(f"   User: {result.user.name}, {result.user.age}")
    print(
        f"   Address: {result.address.street}, {result.address.city}, {result.address.zip}"
    )

    # Complex model with enums and lists
    print("\n3. Complex model with enums and lists:")
    start = time.time()
    project = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": """Create a project with 2 tasks:
                1. A high priority API task (3 hours, tags: backend, api)
                2. A low priority docs task (1 hour, tags: docs)
                Project name: Backend API, status: active""",
            }
        ],
        response_model=Project,
    )
    elapsed = time.time() - start
    print(f"   Project: {project.name} ({project.status})")
    for task in project.tasks:
        print(
            f"   - {task.title}: {task.priority.value}, {task.estimated_hours}h, {task.tags}"
        )

    # Token usage
    if hasattr(project, "_raw_response") and hasattr(project._raw_response, "usage"):
        usage = project._raw_response.usage
        print(
            f"   Tokens: {usage.prompt_tokens} in, {usage.completion_tokens} out, {usage.total_tokens} total"
        )
    print(f"   Time: {elapsed:.2f}s")


def run_anthropic_example():
    """Run TOON mode with Anthropic Claude."""
    try:
        from anthropic import Anthropic
    except ImportError:
        print("\n[Skipping Anthropic - package not installed]")
        return

    import instructor
    from instructor import Mode

    print("\n" + "=" * 60)
    print("Anthropic TOON Mode (ANTHROPIC_TOON)")
    print("=" * 60)

    client = instructor.from_anthropic(Anthropic(), mode=Mode.ANTHROPIC_TOON)

    # Simple extraction
    print("\n1. Simple extraction:")
    user = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Extract: John is 30 and his email is john@example.com",
            }
        ],
        response_model=User,
    )
    print(f"   User: {user.name}, {user.age}, {user.email}")

    # Nested model
    print("\n2. Nested model:")
    result = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Extract: Alice lives at 123 Main St, New York, 10001. She is alice@test.com and 25 years old.",
            }
        ],
        response_model=UserWithAddress,
    )
    print(f"   User: {result.user.name}, {result.user.age}")
    print(
        f"   Address: {result.address.street}, {result.address.city}, {result.address.zip}"
    )


def compare_modes():
    """Compare token usage between TOON and other modes."""
    from openai import OpenAI
    import instructor
    from instructor import Mode

    print("\n" + "=" * 60)
    print("Mode Comparison (Token Usage)")
    print("=" * 60)

    modes = [Mode.TOON, Mode.JSON, Mode.MD_JSON, Mode.TOOLS]
    results = []

    for mode in modes:
        try:
            client = instructor.from_openai(OpenAI(), mode=mode)
            start = time.time()
            project = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": "Create a project 'API' with status active, 1 high priority task: 'Build API', 2 hours, tags: backend",
                    }
                ],
                response_model=Project,
            )
            elapsed = time.time() - start

            usage = getattr(project, "_raw_response", None)
            if usage and hasattr(usage, "usage"):
                total = usage.usage.total_tokens
            else:
                total = 0

            results.append({"mode": mode.value, "tokens": total, "time": elapsed})
            print(f"  {mode.value:<20} {total:>6} tokens  {elapsed:.2f}s")
        except Exception as e:
            print(f"  {mode.value:<20} FAILED: {e}")

    if results and results[0]["tokens"] > 0:
        baseline = results[0]["tokens"]
        print(f"\n  Token savings vs JSON/MD_JSON/TOOLS:")
        for r in results[1:]:
            if r["tokens"] > 0:
                savings = (r["tokens"] - baseline) / r["tokens"] * 100
                print(f"    {r['mode']} saves {savings:.1f}% vs TOON")


def main():
    print("=" * 60)
    print("TOON Mode Examples")
    print("=" * 60)

    run_openai_example()
    run_anthropic_example()
    compare_modes()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
