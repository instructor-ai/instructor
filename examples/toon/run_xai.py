"""TOON Mode Example with xAI

Demonstrates TOON mode with xAI Grok models.

Usage:
    pip install 'instructor[toon]' xai-sdk
    export XAI_API_KEY=your_key
    python run_xai.py
"""

from pydantic import BaseModel


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
        from xai_sdk.sync.client import Client
    except ImportError:
        print("Install xai-sdk: pip install xai-sdk")
        return

    import instructor
    from instructor import Mode

    print("=" * 60)
    print("xAI TOON Mode (XAI_TOON)")
    print("=" * 60)

    client = instructor.from_xai(Client(), mode=Mode.XAI_TOON)

    print("\n1. Simple extraction:")
    user = client.chat.completions.create(
        model="grok-code-fast-1",
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
        model="grok-code-fast-1",
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
