"""
This is a canonical starter example for using `instructor` with OpenAI models.

It demonstrates:
1.  `instructor.from_provider("openai/<model_name>")` for quick client setup.
2.  Defining a simple Pydantic `response_model`.
3.  Making a basic `.chat.completions.create(...)` call with `response_model`.
"""
import instructor
from pydantic import BaseModel, Field

# 1. Initialize the `instructor` client using `from_provider`
#    This sets up the OpenAI client with `instructor`'s patching automatically.
#    Replace "gpt-5.2" with your desired OpenAI model.
client = instructor.from_provider("openai/gpt-5.2")

# 2. Define a simple Pydantic `response_model`
#    The LLM's response will be automatically parsed and validated into this model.
class UserDetail(BaseModel):
    name: str = Field(description="The full name of the user")
    age: int = Field(description="The age of the user in years")
    occupation: str = Field(description="The user's current occupation")

# 3. Make a basic chat completion call with the `response_model`
#    Instructor ensures the LLM's response conforms to the UserDetail schema.
try:
    user_data = client.chat.completions.create(
        model="gpt-5.2",  # Ensure this matches the model in from_provider if specific
        response_model=UserDetail,
        messages=[
            {
                "role": "user",
                "content": "Extract the details for John Doe, who is 30 years old and works as a software engineer.",
            }
        ],
    )

    print("Successfully extracted user data:")
    print(f"Name: {user_data.name}")
    print(f"Age: {user_data.age}")
    print(f"Occupation: {user_data.occupation}")

except Exception as e:
    print(f"An error occurred: {e}")
    print("Please ensure your OPENAI_API_KEY environment variable is set.")

# To run this example:
# 1. Ensure you have an OPENAI_API_KEY set in your environment variables.
# 2. Run: python examples/openai/run.py
