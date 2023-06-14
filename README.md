# Seamless Integration with OpenAI and Pydantic: A Powerful Duo for Output Parsing

Today, OpenAI introduced a Function Call API so we're going to dive into a much more structured and efficient way of handling output parsing when interacting with OpenAI. This method leverages the robustness of the Pydantic library in tandem with the recent improvements in OpenAI's API.

Historically, dealing with output parsing, especially with JSON responses, has been fraught with complexities. Ensuring the extracted data adheres to a specific schema or matches certain function calls often involves writing intricate and cumbersome error-checking code. Add to this the vagaries of AI and you often end up reasking and hoping it does a better job.

However, Pydantic, a Python library that provides data validation through Python type annotations, comes to the rescue! And when combined with OpenAI's new function call capabilities, it allows us to handle output parsing in a much more structured and reliable way with a much better developer experience.

## The Power of Pydantic

Pydantic is a Python library that brings type checking, validation, and error handling to the forefront. By making use of Python type annotations, Pydantic allows you to define data models, validate input data against these models, and receive detailed error messages when data fails validation. This ensures that your data adheres to the correct types, constraints, and formats you specify.

But why Pydantic? Pydantic offers several key benefits:

**Type checking:** Pydantic uses Python type annotations to ensure the data you work with adheres to the correct types. This means less time debugging type-related issues and more confidence in the integrity of your data.

**Validation:** Pydantic allows you to apply additional validation rules to your data models. These could be simple constraints, like numerical ranges, or more complex custom validation functions.

**Error handling:** When validation fails, Pydantic raises detailed exceptions. This gives you a clear understanding of what's gone wrong, making it easier to correct mistakes.

**Ease of use:** Pydantic's data models are just Python classes. You define your data models with familiar Python type annotations, making Pydantic intuitive and easy to use.

**Advanced Features:** Pydantic supports more advanced features like nested models, recursive models, and models with generics. This makes it a flexible and powerful tool for managing complex data.

And when combined with the recent function call capabilities from OpenAI, it brings structured data handling to a whole new level!

## Embracing OpenAI Function Calls

The new function call capabilities introduced by OpenAI mark a significant shift in the way we interact with the OpenAI API. Instead of hoping that a chat message would parse correctly to JSON, we can now specify function calls and their expected inputs. This makes our conversation with the AI more structured and predictable.

Here's where it gets even more interesting. By integrating Pydantic with OpenAI function calls, we can streamline the process of validating the output from OpenAI and handling it in our Python functions. This allows us to interact with the AI in a much more robust and efficient manner.

Let's dive into how we can do this.

## Part 1: Harnessing OpenAI Function Calls with Pydantic

The crux of this approach lies in a simple decorator that handles the mapping between OpenAI function calls and Python functions. This decorator takes care of the input validation, the execution of the function, and the generation of the schema used for the OpenAI function call. Here's how it looks:

```python
@openai_function
def sum(a:int, b:int) -> int:
    """Sum description adds a + b"""
    return a + b
```

In this example, we define a simple function that adds two numbers. We then decorate it with `@openai_function` which takes care of generating the schema for this function and validating the inputs and outputs.

Once we've defined our function, we can interact with the OpenAI API as usual, using the function's schema to guide the conversation:

```python
completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        temperature=0,
        functions=[sum.openai_schema],
        messages=[
            {
                "role": "system",
                "content": "You must use the `sum` function instead of adding yourself.",
            },
            {
                "role": "user",
                "content": "What is 6+3 use the `sum` function",
            },
        ],
    )

result = sum.from_response(completion)
print(result)  # 9
```

Here, we use sum.openai_schema to provide the schema for our function call. This ensures that the AI understands what function to call and what parameters to pass. After the completion is returned, we use sum.from_response(completion) to extract the result from the completion, validate it against our Pydantic model, and return it.

## Part 2: Leveraging OpenAISchema for Data Extraction

Often, we are interested in parsing the output of an OpenAI conversation to extract specific data without making an actual function call. In these cases, we can make use of our OpenAISchema class to define a schema that matches the data we want to extract. Let's look at an example:

```python
class UserDetails(OpenAISchema):
    """User Details"""
    name: str = Field(..., description="User's name")
    age: int = Field(..., description="User's age")

completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=[
        {"role": "system", "content": "I'm going to ask for user details. Use UserDetails to parse this data."},
        {"role": "user", "content": "My name is John Doe and I'm 30 years old."},
    ],
)

user_details = UserDetails.from_response(completion)
print(user_details)  # UserDetails(name="John Doe", age=30)
```

In this example, we define a Pydantic model that represents the data we want to extract. Then, we use UserDetails.from_response(completion) to extract and validate the data from the completion.

## Light, Efficient, and Effective

The key to this approach is its simplicity and efficiency. We make use of just a few lines of Python code to manage input validation, output parsing, and interaction with the OpenAI API. This code is so light that it's better to copy and paste it rather than installing a whole new package.

This methodology cuts down on unnecessary abstraction, letting you stay in control and fully understand the interaction with the underlying API. It's an elegant and powerful solution for working with the OpenAI API in a structured and reliable way, proving you can have your cake and eat it too!
