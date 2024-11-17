import os
import instructor
from mistralai import Mistral
from pydantic import BaseModel
from rich import print
import asyncio
from typing import List

class User(BaseModel):
    name: str
    age: int
    bio: str = ""

async def test_async_streaming():
    print("[bold blue]Testing Async Streaming[/bold blue]")
    try:
        mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        client = instructor.from_mistral(mistral_client, mode=instructor.Mode.MISTRAL_TOOLS, use_async=True)

        user = await client.create(
            model="mistral-large-latest",
            messages=[
                {"role": "user", "content": "Create a user profile for Jason, age 25"},
            ],
            response_model=User
        )
        print(f"\nAsync Result: {user}")
        print("\n[green]✓[/green] Async streaming test completed")

    except Exception as e:
        print(f"[red]✗[/red] Error in async streaming test: {str(e)}")

def test_basic():
    print("\n[bold blue]Testing Basic Usage[/bold blue]")
    try:
        mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        client = instructor.from_mistral(mistral_client, mode=instructor.Mode.MISTRAL_TOOLS)

        user = client.create(
            model="mistral-large-latest",
            messages=[
                {"role": "user", "content": "Create a user profile for Jason, age 25, with a detailed bio"},
            ],
            response_model=User
        )
        print(f"\nBasic Result: {user}")
        print("\n[green]✓[/green] Basic test completed")

    except Exception as e:
        print(f"[red]✗[/red] Error in basic test: {str(e)}")

def test_multiple_users():
    print("\n[bold blue]Testing Multiple Users[/bold blue]")
    try:
        mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        client = instructor.from_mistral(mistral_client, mode=instructor.Mode.MISTRAL_TOOLS)

        users = client.create(
            model="mistral-large-latest",
            messages=[
                {"role": "user", "content": """
                    Extract users:
                    1. Jason is 25 years old
                    2. Sarah is 30 years old
                    3. Mike is 28 years old
                """}
            ],
            response_model=List[User]
        )

        print("\nMultiple Users Result:")
        for user in users:
            print(f"User: {user}")

        print("\n[green]✓[/green] Multiple users test completed")

    except Exception as e:
        print(f"[red]✗[/red] Error in multiple users test: {str(e)}")

if __name__ == "__main__":
    print("[bold yellow]Starting Mistral Integration Tests[/bold yellow]\n")

    # Run sync tests
    test_basic()
    test_multiple_users()

    # Run async test
    asyncio.run(test_async_streaming())

    print("\n[bold green]All tests completed[/bold green]")
