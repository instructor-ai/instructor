import os
import instructor
from anthropic import Anthropic
from pydantic import BaseModel
from rich import print
from typing import List

class User(BaseModel):
    name: str
    age: int
    bio: str = ""

def test_basic_streaming():
    print("[bold blue]Testing Basic Streaming[/bold blue]")
    try:
        client = instructor.from_anthropic(Anthropic())

        # Test partial streaming
        print("\nTesting Partial Streaming:")
        for partial_user in client.messages.create_partial(
            model="claude-3-opus-20240229",
            messages=[
                {"role": "user", "content": "Create a user profile for Jason, age 25, with a detailed bio"},
            ],
            response_model=User,
        ):
            print(f"Partial State: {partial_user}")

        print("\n[green]✓[/green] Partial streaming test completed")

    except Exception as e:
        print(f"[red]✗[/red] Error in streaming test: {str(e)}")

def test_iterable_streaming():
    print("\n[bold blue]Testing Iterable Streaming[/bold blue]")
    try:
        client = instructor.from_anthropic(Anthropic())

        # Test iterable streaming
        users = client.messages.create_iterable(
            model="claude-3-opus-20240229",
            messages=[
                {"role": "user", "content": """
                    Extract users:
                    1. Jason is 25 years old
                    2. Sarah is 30 years old
                    3. Mike is 28 years old
                """},
            ],
            response_model=User,
        )

        print("\nTesting Iterable Streaming:")
        for user in users:
            print(f"Extracted User: {user}")

        print("\n[green]✓[/green] Iterable streaming test completed")

    except Exception as e:
        print(f"[red]✗[/red] Error in iterable test: {str(e)}")

if __name__ == "__main__":
    print("[bold yellow]Starting Anthropic Streaming Tests[/bold yellow]\n")
    test_basic_streaming()
    test_iterable_streaming()
    print("\n[bold green]All tests completed[/bold green]")
