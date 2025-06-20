---
authors:
- jxnl
categories:
- LLM Techniques
comments: true
date: 2023-11-13
description: "Master Python asyncio.gather and asyncio.as_completed for efficient concurrent LLM processing with Instructor. Learn async programming patterns, rate limiting, and performance optimization for AI applications."
draft: false
slug: learn-async
tags:
- asyncio
- asyncio.gather
- asyncio.as_completed
- OpenAI
- Python
- data processing
- async programming
- concurrent processing
- LLM optimization
---

# Mastering Python asyncio.gather and asyncio.as_completed for LLM Processing

Learn how to use Python's `asyncio.gather` and `asyncio.as_completed` for efficient concurrent processing of Large Language Models (LLMs) with Instructor. This comprehensive guide covers async programming patterns, rate limiting strategies, and performance optimization techniques.

<!-- more -->

!!! notes "Complete Example Code"

    You can find the complete working example on [GitHub](https://github.com/jxnl/instructor/blob/main/examples/learn-async/run.py)

## Understanding asyncio.gather vs asyncio.as_completed

Python's `asyncio` library provides two powerful methods for concurrent execution:

- **`asyncio.gather`**: Executes all tasks concurrently and returns results in the same order as input
- **`asyncio.as_completed`**: Returns results as they complete, regardless of input order

Both methods significantly outperform sequential processing, but they serve different use cases.

## Complete Setup: Async LLM Processing

Here's a complete, self-contained example showing how to set up async processing with Instructor:

```python
import asyncio
import time
from typing import List
import instructor
from pydantic import BaseModel
from openai import AsyncOpenAI

# Set up the async client with Instructor
client = instructor.from_openai(AsyncOpenAI())

class Person(BaseModel):
    name: str
    age: int
    occupation: str

async def extract_person(text: str) -> Person:
    """Extract person information from text using LLM."""
    return await client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=Person,
        messages=[{"role": "user", "content": f"Extract person info: {text}"}]
    )

# Sample dataset
dataset = [
    "John Smith is a 30-year-old software engineer",
    "Sarah Johnson is a 25-year-old data scientist",
    "Mike Davis is a 35-year-old product manager",
    "Lisa Wilson is a 28-year-old UX designer",
    "Tom Brown is a 32-year-old DevOps engineer",
    "Emma Garcia is a 27-year-old frontend developer",
    "David Lee is a 33-year-old backend developer"
]
```

## Method 1: Sequential Processing (Baseline)

```python
async def sequential_processing() -> List[Person]:
    """Process items one by one - slowest method."""
    start_time = time.time()
    persons = []

    for text in dataset:
        person = await extract_person(text)
        persons.append(person)
        print(f"Processed: {person.name}")

    end_time = time.time()
    print(f"Sequential processing took: {end_time - start_time:.2f} seconds")
    return persons

# Run sequential processing
# persons = await sequential_processing()
```

## Method 2: asyncio.gather - Concurrent Processing

```python
async def gather_processing() -> List[Person]:
    """Process all items concurrently and return in order."""
    start_time = time.time()

    # Create tasks for all items
    tasks = [extract_person(text) for text in dataset]

    # Execute all tasks concurrently
    persons = await asyncio.gather(*tasks)

    end_time = time.time()
    print(f"asyncio.gather took: {end_time - start_time:.2f} seconds")

    # Results maintain original order
    for person in persons:
        print(f"Processed: {person.name}")

    return persons

# Run gather processing
# persons = await gather_processing()
```

## Method 3: asyncio.as_completed - Streaming Results

```python
async def as_completed_processing() -> List[Person]:
    """Process items concurrently and handle results as they complete."""
    start_time = time.time()
    persons = []

    # Create tasks for all items
    tasks = [extract_person(text) for text in dataset]

    # Process results as they complete
    for task in asyncio.as_completed(tasks):
        person = await task
        persons.append(person)
        print(f"Completed: {person.name}")

    end_time = time.time()
    print(f"asyncio.as_completed took: {end_time - start_time:.2f} seconds")
    return persons

# Run as_completed processing
# persons = await as_completed_processing()
```

## Method 4: Rate-Limited Processing with Semaphores

```python
async def rate_limited_extract_person(text: str, semaphore: asyncio.Semaphore) -> Person:
    """Extract person info with rate limiting."""
    async with semaphore:
        return await extract_person(text)

async def rate_limited_gather(concurrency_limit: int = 3) -> List[Person]:
    """Process items with controlled concurrency using asyncio.gather."""
    start_time = time.time()

    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(concurrency_limit)

    # Create rate-limited tasks
    tasks = [rate_limited_extract_person(text, semaphore) for text in dataset]

    # Execute with rate limiting
    persons = await asyncio.gather(*tasks)

    end_time = time.time()
    print(f"Rate-limited gather (limit={concurrency_limit}) took: {end_time - start_time:.2f} seconds")
    return persons

async def rate_limited_as_completed(concurrency_limit: int = 3) -> List[Person]:
    """Process items with controlled concurrency using asyncio.as_completed."""
    start_time = time.time()
    persons = []

    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(concurrency_limit)

    # Create rate-limited tasks
    tasks = [rate_limited_extract_person(text, semaphore) for text in dataset]

    # Process results as they complete
    for task in asyncio.as_completed(tasks):
        person = await task
        persons.append(person)
        print(f"Rate-limited completed: {person.name}")

    end_time = time.time()
    print(f"Rate-limited as_completed (limit={concurrency_limit}) took: {end_time - start_time:.2f} seconds")
    return persons

# Run rate-limited processing
# persons = await rate_limited_gather(concurrency_limit=2)
# persons = await rate_limited_as_completed(concurrency_limit=2)
```

## Advanced Patterns

### 1. Error Handling with asyncio.gather

```python
async def robust_gather_processing() -> List[Person]:
    """Process items with error handling."""
    tasks = [extract_person(text) for text in dataset]

    # Execute with error handling
    results = await asyncio.gather(*tasks, return_exceptions=True)

    persons = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Error processing item {i}: {result}")
        else:
            persons.append(result)

    return persons
```

### 2. Timeout Handling

```python
async def timeout_gather_processing(timeout_seconds: float = 10.0) -> List[Person]:
    """Process items with timeout."""
    tasks = [extract_person(text) for text in dataset]

    try:
        persons = await asyncio.wait_for(asyncio.gather(*tasks), timeout=timeout_seconds)
        return persons
    except asyncio.TimeoutError:
        print(f"Processing timed out after {timeout_seconds} seconds")
        return []
```

### 3. Progress Tracking

```python
async def progress_tracking_processing() -> List[Person]:
    """Process items with progress tracking."""
    persons = []
    total_items = len(dataset)
    completed = 0

    tasks = [extract_person(text) for text in dataset]

    for task in asyncio.as_completed(tasks):
        person = await task
        persons.append(person)
        completed += 1
        print(f"Progress: {completed}/{total_items} ({completed/total_items*100:.1f}%)")

    return persons
```

### 4. Batch Processing with Chunking

```python
async def chunked_processing(chunk_size: int = 3) -> List[Person]:
    """Process items in chunks to manage memory and rate limits."""
    all_persons = []

    # Process in chunks
    for i in range(0, len(dataset), chunk_size):
        chunk = dataset[i:i + chunk_size]
        print(f"Processing chunk {i//chunk_size + 1}")

        tasks = [extract_person(text) for text in chunk]
        chunk_results = await asyncio.gather(*tasks)
        all_persons.extend(chunk_results)

    return all_persons
```

## Performance Comparison

Here are typical performance results when processing 7 items:

| Method | Execution Time | Concurrency | Use Case |
|--------|---------------|-------------|----------|
| Sequential | 6.17 seconds | 1 | Baseline |
| asyncio.gather | 0.85 seconds | 7 | Fast processing, ordered results |
| asyncio.as_completed | 0.95 seconds | 7 | Streaming results |
| Rate-limited gather | 3.04 seconds | 2 | API-friendly |
| Rate-limited as_completed | 3.26 seconds | 2 | Streaming + rate limiting |

## When to Use Each Method

### Use asyncio.gather when:
- You need results in the same order as input
- All tasks must complete successfully
- You want the fastest possible execution
- Memory usage isn't a concern

### Use asyncio.as_completed when:
- You want to process results as they arrive
- Order doesn't matter
- You're streaming data to clients
- You want to handle large datasets efficiently

### Use rate limiting when:
- Working with API rate limits
- Being respectful to external services
- Managing resource consumption
- Building production applications

## Best Practices

### 1. Error Handling
```python
async def safe_processing():
    try:
        persons = await asyncio.gather(*tasks, return_exceptions=True)
        return [p for p in persons if not isinstance(p, Exception)]
    except Exception as e:
        print(f"Processing failed: {e}")
        return []
```

### 2. Resource Management
```python
async def resource_aware_processing():
    # Limit concurrent connections
    connector = aiohttp.TCPConnector(limit=10)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Your processing code here
        pass
```

### 3. Monitoring and Logging
```python
import logging

async def monitored_processing():
    logger = logging.getLogger(__name__)
    start_time = time.time()

    persons = await asyncio.gather(*tasks)

    end_time = time.time()
    logger.info(f"Processed {len(persons)} items in {end_time - start_time:.2f}s")
    return persons
```

## Complete Example: Main Function

```python
async def main():
    """Run all processing methods and compare performance."""
    print("=== Python asyncio.gather and asyncio.as_completed Performance Test ===\n")

    # Test different methods
    methods = [
        ("Sequential", sequential_processing),
        ("asyncio.gather", gather_processing),
        ("asyncio.as_completed", as_completed_processing),
        ("Rate-limited gather", lambda: rate_limited_gather(2)),
        ("Rate-limited as_completed", lambda: rate_limited_as_completed(2))
    ]

    results = {}
    for name, method in methods:
        print(f"\n--- Testing {name} ---")
        try:
            persons = await method()
            results[name] = len(persons)
            print(f"Successfully processed {len(persons)} items")
        except Exception as e:
            print(f"Error with {name}: {e}")

    print(f"\n=== Results Summary ===")
    for name, count in results.items():
        print(f"{name}: {count} items processed")

# Run the complete example
if __name__ == "__main__":
    asyncio.run(main())
```

## Key Takeaways

1. **asyncio.gather** is fastest for ordered results
2. **asyncio.as_completed** is best for streaming and large datasets
3. **Rate limiting** is essential for production applications
4. **Error handling** should be implemented for robustness
5. **Monitoring** helps optimize performance

## Related Resources

- [Python asyncio Documentation](https://docs.python.org/3/library/asyncio.html)
- [Real Python Async IO Tutorial](https://realpython.com/async-io-python/)
- [Instructor Documentation](https://python.useinstructor.com)
- [OpenAI Async API Guide](https://platform.openai.com/docs/guides/async)

---

**Next Steps**: Learn about [error handling patterns](../concepts/error_handling.md) or explore [rate limiting with tenacity](../concepts/retrying.md) for production applications.