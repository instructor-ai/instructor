"""TOON Mode Example with Google GenAI

Demonstrates TOON mode with Google Gemini models via the new GenAI API.

Usage:
    pip install 'instructor[toon]' google-genai
    export GOOGLE_API_KEY=your_key
    python run_genai.py
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
        from google.genai import Client
    except ImportError:
        print("Install google-genai: pip install google-genai")
        return

    import instructor
    from instructor import Mode

    print("=" * 60)
    print("Google GenAI TOON Mode (GENAI_TOON)")
    print("=" * 60)

    client = instructor.from_genai(Client(), mode=Mode.GENAI_TOON)

    print("\n1. Simple extraction:")
    user = client.chat.completions.create(
        model="gemini-2.5-flash",
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
        model="gemini-2.5-flash",
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
