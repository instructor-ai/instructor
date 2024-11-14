from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/",
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "Users",
            "description": "Correctly extracted `Users` with all the required parameters with correct types",
            "parameters": {
                "$defs": {
                    "User": {
                        "properties": {
                            "name": {"title": "Name", "type": "string"},
                            "age": {"title": "Age", "type": "integer"},
                        },
                        "required": ["name", "age"],
                        "title": "User",
                        "type": "object",
                    }
                },
                "properties": {
                    "users": {
                        "items": {"$ref": "#/$defs/User"},
                        "title": "Users",
                        "type": "array",
                    }
                },
                "required": ["users"],
                "type": "object",
            },
        },
    }
]

response = client.chat.completions.create(
    model="gemini-1.5-flash",
    messages=[
        {
            "role": "user",
            "content": "Ivan is 20 and lives in Singapore. His friend Darren lives in Malaysia and is the same age",
        }
    ],
    tools=tools,
    tool_choice="auto",
)

print(response)
