---
authors:
  - jxnl
  - ivanleomk
categories:
  - instructor
comments: true
date: 2025-05-08
description: Switch between different models and providers with a single string!
draft: false
tags:
  - LLMs
  - Instructor
---

# One String to Rule Them All

With instructor 1.8.0, you can now switch between any LLM providers with a single line of code.

This new unified provider interface lets you initialise any supported provider (Eg. OpenAI, Anthropic, Google ) without modifying any other bits of your existing code.

This eliminates mountains of boilerplate code and dependency management and allows you to

1. **Stop fiddling with provider-specific setup code** - forget juggling different client libraries and learning each provider's quirks

2. **Conduct rapid experimentats** - test models across providers without rewriting your code, letting you find the best model for your specific use case

3. **Simplify dependency management** - we handle the compatibility challenges between each client and all the tough bits in between

This frees you to focus on what matters - the business logic helping you to build a better LLM application.

<!-- more -->

## String Initialisation

Let's see this in action below!

```python
import instructor
from pydantic import BaseModel
from typing import Iterable

# Define your data structure
class Person(BaseModel):
    name: str
    age: int

# Connect to any provider with a single line
client = instructor.from_provider("google/gemini-2.0-flash")

# Extract structured data
response = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Alice is 30 and Bob is 25.",
        }
    ],
    response_model=Iterable[Person],
)

for person in response:
    print(f"Name: {person.name}, Age: {person.age}")
# Output:
# Name: Alice, Age: 30
# Name: Bob, Age: 25
```

Switching providers is as simple as changing the string:

```python
# OpenAI
client = instructor.from_provider("openai/gpt-4.1")

# Anthropic (with version date)
client = instructor.from_provider("anthropic/claude-3-5-haiku-20241022")
```

With the unified provider interface, you can now easily benchmark different models on the same task. This is crucial when you need to:

1. Compare response quality across different providers
2. Test which model gives the best structured extraction results
3. Optimize for speed vs. accuracy tradeoffs
4. Run A/B tests between providers without code refactoring

Instead of maintaining separate codebases for each provider or complex switching logic, you can focus on what matters: finding the optimal model for your specific use case.

### Async Support

When building production applications that need to remain responsive, asynchronous processing is essential.

Instructor's unified provider interface supports this workflow with a simple `async_client` keyword during initialization.

```python
client = instructor.from_provider("openai/gpt-4.1", async_client=True)
```

The async implementation works particularly well for web servers, batch processing jobs, or any scenario where you need to extract structured data without blocking your application's main thread.

Here's how you can implement it:

```python
import instructor
from pydantic import BaseModel
import asyncio

class UserProfile(BaseModel):
    name: str
    country: str

async def get_user_profile():
    # Initialise an asynchronous client
    async_client = instructor.from_provider(
        "openai/gpt-4.1-mini",
        async_client=True
    )

    # Extract data asynchronously
    profile = await async_client.chat.completions.create(
        messages=[{"role": "user", "content": "Extract: Maria lives in Spain."}],
        response_model=UserProfile
    )
    print(f"Name: {profile.name}, Country: {profile.country}")

if __name__ == "__main__":
    asyncio.run(get_user_profile())
```

### Provider Specific Parameters

Some providers require additional parameters for optimal performance.

Rather than hiding these options, Instructor allows you to pass them directly through the from_provider function:

```python
# Anthropic requires max tokens
client = instructor.from_provider(
    "anthropic/claude-3-sonnet-20240229",
    max_tokens=1024
)
```

If you'd like to change this parameter down the line, you can just do so by setting it on the `client.chat.completions.create` function again.

### Type Completion

To make it easy for you to find the right model string, we now ship with auto-complete for these new model-provider initialisation strings.

This is automatically provided for you out of the box when you use the new `from_provider` method as seen below.

![](./img/instructor-autocomplete.png)

Say bye to fiddling around with messy model versioning and get cracking to working on your business logic instead!

## Get Started Today

Upgrade to the latest version and make sure you've installed `1.8.0`.

```
pip install --upgrade instructor
```

Check out our [full documentation](https://python.useinstructor.com/) for more examples and supported providers.
