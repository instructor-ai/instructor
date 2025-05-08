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
client = instructor.from_provider("google/gemini-1.5-flash")

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
client = instructor.from_provider("openai/gpt-4o-mini")

# Anthropic (with version date)
client = instructor.from_provider("anthropic/claude-3-haiku-20240307")
```

Experimentation becomes significantly easier with this unified interface.

When you want to test how different models perform on the same task, you simply change the provider string rather than rewriting chunks of your codebase.

This makes writing tests and experiments much faster allowing you to spend less time maintaining boilerplate code and more time experimenting to find the optimal model and prompt for your use case.

Behind the scenes, Instructor handles all the complexities of different client libraries. This simplified dependency management means you don't need to worry about compatibility issues or keeping up with the latest changes to each provider's SDK.

### Async Support

When building production applications that need to remain responsive, asynchronous processing is essential. Instructor's unified provider interface supports this workflow with a simple `async_client` keyword during initialization.

```python
client = instructor.from_provider("openai/gpt-4o-mini", async_client=True)
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
    # Initialize an asynchronous client
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
