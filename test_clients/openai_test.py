from typing import List, Iterator
import instructor
from pydantic import BaseModel
import openai
from rich import print

# Enable instructor patch
client = instructor.patch(openai.OpenAI())

class UserInfo(BaseModel):
    name: str
    age: int
    hobbies: List[str]

class PartialUserInfo(BaseModel):
    name: str = ""
    age: int = 0
    hobbies: List[str] = []

def test_basic():
    """Test basic structured output"""
    try:
        user = client.chat.completions.create(
            model="gpt-3.5-turbo",
            response_model=UserInfo,
            messages=[
                {"role": "user", "content": "Extract: John is 30 years old and enjoys reading, hiking, and photography."}
            ]
        )
        print("[green]✓ Basic test successful:[/green]", user)
        return True
    except Exception as e:
        print("[red]✗ Basic test failed:[/red]", str(e))
        return False

def test_streaming():
    """Test streaming support"""
    try:
        user_stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            response_model=UserInfo,
            messages=[
                {"role": "user", "content": "Extract: John is 30 years old and enjoys reading, hiking, and photography."}
            ],
            stream=True
        )
        print("[green]✓ Streaming test:[/green]")
        for chunk in user_stream:
            print(f"  Chunk: {chunk}")
        return True
    except Exception as e:
        print("[red]✗ Streaming test failed:[/red]", str(e))
        return False

def test_partial_streaming():
    """Test partial streaming support"""
    try:
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            response_model=PartialUserInfo,
            messages=[
                {"role": "user", "content": "Extract: John is 30 years old and enjoys reading, hiking, and photography."}
            ],
            stream=True,
            partial=True
        )
        print("[green]✓ Partial streaming test:[/green]")
        for partial in stream:
            print(f"  Partial: {partial}")
        return True
    except Exception as e:
        print("[red]✗ Partial streaming test failed:[/red]", str(e))
        return False

def test_iterable():
    """Test iterable response"""
    class UserList(BaseModel):
        users: List[UserInfo]

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            response_model=UserList,
            messages=[
                {"role": "user", "content": """Extract multiple users:
                John is 30 years old and enjoys reading, hiking, and photography.
                Mary is 25 and likes painting, cooking, and gardening."""}
            ]
        )
        print("[green]✓ Iterable test successful:[/green]", response)
        return True
    except Exception as e:
        print("[red]✗ Iterable test failed:[/red]", str(e))
        return False

if __name__ == "__main__":
    print("\n[bold]Testing OpenAI Integration[/bold]\n")
    results = {
        "Basic": test_basic(),
        "Streaming": test_streaming(),
        "Partial Streaming": test_partial_streaming(),
        "Iterable": test_iterable()
    }

    print("\n[bold]Summary:[/bold]")
    for test, passed in results.items():
        status = "[green]✓ Passed[/green]" if passed else "[red]✗ Failed[/red]"
        print(f"{test}: {status}")
