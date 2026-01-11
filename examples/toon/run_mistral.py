"""TOON Mode Example with Mistral

Demonstrates TOON mode with Mistral AI models.

Usage:
    pip install 'instructor[toon]' mistralai
    export MISTRAL_API_KEY=your_key
    python run_mistral.py
"""

from pydantic import BaseModel
import os


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


def main():
    try:
        from mistralai import Mistral
    except ImportError:
        print("Install mistralai: pip install mistralai")
        return

    import instructor
    from instructor import Mode

    print("=" * 60)
    print("Mistral TOON Mode (MISTRAL_TOON)")
    print("=" * 60)

    client = instructor.from_mistral(
        Mistral(
            # NOTE: for some reason this wasnt pulling key on it's own, not sure
            # if it was intended to be this way.
            api_key=os.environ.get("MISTRAL_API_KEY"),
        ),
        mode=Mode.MISTRAL_TOON,
    )

    print("\n1. Simple extraction:")
    user = client.chat.completions.create(
        model="devstral-small-latest",
        messages=[
            {
                "role": "user",
                "content": "Extract: John is 30 and his email is john@example.com",
            }
        ],
        response_model=User,
    )
    print(f"   User: {user.name}, {user.age}, {user.email}")

    print("\n2. Nested model:")
    result = client.chat.completions.create(
        model="devstral-small-latest",
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

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
