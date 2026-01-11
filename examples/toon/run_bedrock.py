"""TOON Mode Example with AWS Bedrock

Demonstrates TOON mode with AWS Bedrock models (Claude, Llama, etc.).

Usage:
    pip install 'instructor[toon]' boto3
    # Configure AWS credentials via environment or ~/.aws/credentials
    python run_bedrock.py
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
        import boto3
    except ImportError:
        print("Install boto3: pip install boto3")
        return

    import instructor
    from instructor import Mode

    print("=" * 60)
    print("AWS Bedrock TOON Mode (BEDROCK_TOON)")
    print("=" * 60)

    bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
    client = instructor.from_bedrock(bedrock, mode=Mode.BEDROCK_TOON)

    print("\n1. Simple extraction:")
    user = client.chat.completions.create(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
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
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
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
