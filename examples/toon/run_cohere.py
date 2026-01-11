"""TOON Mode Example with Cohere

Demonstrates TOON mode with Cohere Command models.

Usage:
    pip install 'instructor[toon]' cohere
    export COHERE_API_KEY=your_key
    python run_cohere.py
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
        import cohere
    except ImportError:
        print("Install cohere: pip install cohere")
        return

    import instructor
    from instructor import Mode

    print("=" * 60)
    print("Cohere TOON Mode (COHERE_TOON)")
    print("=" * 60)

    client = instructor.from_cohere(cohere.ClientV2(), mode=Mode.COHERE_TOON)

    print("\n1. Simple extraction:")
    user = client.chat.completions.create(
        model="command-r-plus",
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
        model="command-r-plus",
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
