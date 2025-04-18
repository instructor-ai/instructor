# Batch Processing

This guide covers using batch processing with Instructor to handle multiple extraction requests efficiently.

## Basic Example

```python
import asyncio
from openai import AsyncOpenAI
import instructor
from pydantic import BaseModel
from typing import List

# Define the structure you want to extract
class Person(BaseModel):
    name: str
    age: int

# Patch AsyncOpenAI client
client = instructor.apatch(AsyncOpenAI())

# Async function to extract a person
async def extract_person(text: str) -> Person:
    return await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": text},
        ],
        response_model=Person,
    )

# Batch processing with asyncio.gather
async def process_batch():
    dataset = [
        "My name is John and I am 30 years old",
        "My name is Maria and I am 25 years old",
        "My name is Alex and I am 42 years old",
    ]
    
    # Create tasks for all items
    tasks = [extract_person(text) for text in dataset]
    
    # Process all at once
    all_persons = await asyncio.gather(*tasks)
    
    # Print results
    for person in all_persons:
        print(f"Name: {person.name}, Age: {person.age}")
    
    return all_persons

# Run the batch process
if __name__ == "__main__":
    asyncio.run(process_batch())
```

```
┌───────────────┐            ┌───────────────┐
│ Multiple      │            │ Concurrent    │
│ Text Items    │  Extract   │ Processing    │
│               │ ─────────> │ with Asyncio  │
└───────────────┘            └───────────────┘
```

## Rate-Limited Batch Processing

Control concurrent requests to avoid hitting rate limits:

```python
import asyncio
from openai import AsyncOpenAI
import instructor
from pydantic import BaseModel
from typing import List

class Person(BaseModel):
    name: str
    age: int

# Patch AsyncOpenAI client
client = instructor.apatch(AsyncOpenAI())

# Batch processing with rate limiting
async def process_rate_limited_batch():
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(2)  # Maximum 2 concurrent requests
    
    dataset = [
        "My name is John and I am 30 years old",
        "My name is Maria and I am 25 years old",
        "My name is Alex and I am 42 years old",
        "My name is Sarah and I am 38 years old",
        "My name is Michael and I am 45 years old",
    ]
    
    async def rate_limited_extract(text: str) -> Person:
        async with semaphore:  # Only 2 requests run at once
            print(f"Processing: {text[:15]}...")
            return await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": text}],
                response_model=Person,
            )
    
    # Create tasks for all items
    tasks = [rate_limited_extract(text) for text in dataset]
    
    # Process with rate limiting
    all_persons = await asyncio.gather(*tasks)
    return all_persons

if __name__ == "__main__":
    asyncio.run(process_rate_limited_batch())
```

## Next Steps

- Try [Async Processing](async_processing.md) for more details on async operations
- Learn about [Caching Techniques](caching_techniques.md) to optimize API usage
- Explore [Streaming Lists](../streaming/lists.md) for real-time batch processing