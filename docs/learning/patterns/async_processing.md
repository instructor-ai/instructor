# Async Processing

This guide covers using asynchronous processing with Instructor for efficient concurrent operations.

## Basic Example

```python
import asyncio
from pydantic import BaseModel
import instructor
from openai import AsyncOpenAI

# Define the structure you want to extract
class Person(BaseModel):
    name: str
    age: int
    occupation: str

# Create async function for extraction
async def extract_person(text: str) -> Person:
    client = instructor.from_openai(AsyncOpenAI())
    person = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": text}
        ],
        response_model=Person
    )
    return person

# Process multiple extractions concurrently
async def main():
    texts = [
        "John Smith is a 35-year-old software engineer.",
        "Maria Rodriguez is a 42-year-old doctor.",
        "Alex Johnson is a 28-year-old graphic designer."
    ]
    
    # Process all extractions concurrently
    results = await asyncio.gather(*[extract_person(text) for text in texts])
    
    # Print results
    for person in results:
        print(f"Name: {person.name}, Age: {person.age}, Occupation: {person.occupation}")

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())
```

```
┌───────────────┐            ┌───────────────┐
│ Async         │            │ Concurrent    │
│ Processing    │  Extract   │ Processing    │
│ with          │ ─────────> │ of Multiple   │
│ Instructor    │            │ Extractions   │
└───────────────┘            └───────────────┘
```

## Using Semaphores to Control Concurrency

Semaphores limit how many operations run at once to prevent overwhelming APIs:

```python
import asyncio
from pydantic import BaseModel
import instructor
from openai import AsyncOpenAI

class Person(BaseModel):
    name: str
    age: int
    occupation: str

async def extract_with_semaphore(text: str, sem: asyncio.Semaphore) -> Person:
    # The semaphore limits how many concurrent requests can be active
    async with sem:
        # Only this many requests will run at the same time
        print(f"Processing: {text[:20]}...")
        client = instructor.from_openai(AsyncOpenAI())
        person = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": text}],
            response_model=Person
        )
        return person

async def main():
    # Create texts to process
    texts = [f"Person {i}: Name is John Smith, age is 35, works as engineer." for i in range(20)]
    
    # Create a semaphore that allows 5 concurrent operations
    # This prevents hitting rate limits or using too many resources
    semaphore = asyncio.Semaphore(5)
    
    # Create tasks with the semaphore
    tasks = [extract_with_semaphore(text, semaphore) for text in texts]
    
    # Run all tasks with controlled concurrency
    results = await asyncio.gather(*tasks)
    
    print(f"Processed {len(results)} items")

if __name__ == "__main__":
    asyncio.run(main())
```

## Simple Comparison with Sync Method

```python
import time
import asyncio
from pydantic import BaseModel
import instructor
from openai import OpenAI, AsyncOpenAI

class Person(BaseModel):
    name: str
    age: int
    occupation: str

# Synchronous version - processes one at a time
def sync_process(texts):
    client = instructor.from_openai(OpenAI())
    start_time = time.time()
    
    results = []
    for text in texts:
        result = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": text}],
            response_model=Person
        )
        results.append(result)
    
    elapsed = time.time() - start_time
    print(f"Sync processing took {elapsed:.2f} seconds")
    return results

# Async version with semaphore - processes multiple concurrently
async def async_process(texts):
    client = instructor.from_openai(AsyncOpenAI())
    start_time = time.time()
    
    # Create a semaphore allowing 5 concurrent operations
    sem = asyncio.Semaphore(5)
    
    async def process_single(text):
        async with sem:  # Only 5 requests run at once
            return await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": text}],
                response_model=Person
            )
    
    results = await asyncio.gather(*[process_single(text) for text in texts])
    
    elapsed = time.time() - start_time
    print(f"Async processing took {elapsed:.2f} seconds")
    return results
```

## Next Steps

- Try [Batch Processing](batch_processing.md) to handle multiple items at once
- Learn about [Caching Techniques](caching_techniques.md) to optimize API usage
- Explore [Streaming](../streaming/basics.md) for real-time results