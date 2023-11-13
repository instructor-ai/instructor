---
draft: False
date: 2023-11-13
tags:
  - python
  - batch
  - asyncio
  - async
  - async/await
authors:
  - jxnl
---

# Learning AsyncIO in Python with Instructor

More and more I'm finding instructor as a light weight library that tries to also teach some of the best practices in python. This blog post is a great example of that. Today I'm going to introduce you to some ways of using asyncio in python, applying what we learn to batch processing tasks. We'll explore a script that demonstrates the use of AsyncIO for batch processing tasks, specifically extracting and processing data concurrently.

## Basic Concepts of `asyncio`

`asyncio` is a Python library used for writing concurrent code using the async/await syntax. It's ideal for IO-bound and high-level structured network code. If you've used `OpenAI`'s SDK you'll notice that theres both a `OpenAI()` and `AsyncOpenAI()` class. The `AsyncOpenAI()` class is a subclass of `OpenAI()` processes data asynchronously.

By doing so in the context of serving a web application or batch processing tasks, we can improve the performance of our application by allowing it to handle multiple requests concurrently!

## Batch Processing in Action

Today we'll be looking at an example that demonstrates the use of AsyncIO for batch processing tasks, specifically extracting and processing data concurrently. The script will extract data from a list of texts and process it concurrently using AsyncIO.

```python
import instructor
from pydantic import BaseModel
from openai import AsyncOpenAI

# Enables `response_model` in `create` method
client = instructor.apatch(AsyncOpenAI()) # (1)!

class Person(BaseModel):
    name: str
    age: int


async def extract_person(text: str) -> Person:
    return await client.chat.completions.create( # (2)!
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": text},
        ],
        response_model=Person,
    )
```

1.  We use `instructor.apatch` to patch the `create` method of `AsyncOpenAI` to accept a `response_model` argument. This is because the `create` method of `AsyncOpenAI` does not accept a `response_model` argument without this patch.
2.  We use `await` here to wait for the response from the server before we return the result. This is because `create` returns a coroutine object, not the result of the coroutine.

Notice that now there are `async` and `await` keywords in the function definition. This is because we're using the `asyncio` library to run the function concurrently. Now lets define a batch of texts to process.

```python
dataset = [
        "My name is John and I am 20 years old",
        "My name is Mary and I am 21 years old",
        "My name is Bob and I am 22 years old",
        "My name is Alice and I am 23 years old",
        "My name is Jane and I am 24 years old",
        "My name is Joe and I am 25 years old",
        "My name is Jill and I am 26 years old",
    ]
```

1. **`for` Loop**: Running tasks sequentially.

```python
for text in dataset:
    person = await extract_person(text)
    persons.append(person)
```

Here even tho theres an await we still have to wait for each task to finish before we can start the next one. This is because we're using a for loop to iterate over the dataset. This will be the slowest method of the four we'll be looking at today.

2. **`asyncio.gather`**: Running tasks concurrently.

```python
async def gather()
    tasks_get_persons = [extract_person(text) for text in dataset]
    all_persons = await asyncio.gather(*tasks_get_persons) # (1)!
```

1.  We use `await` here to wait for all the tasks to finish before we assign the result to `all_persons`. This is because `asyncio.gather` returns a coroutine object, not the result of the coroutine. We can also use `asyncio.as_completed` to do this.

Using gather means we want to return everything all at once. This is a great way to speed up our code, but it's not the only way. In particular if we have a large dataset we might not want to wait for everything to finish before we start processing the results. This is where `asyncio.as_completed` comes in.

3. **`asyncio.as_completed`**: Handling tasks as they complete.

```python
async def as_completed()
    all_persons = []
    tasks_get_persons = [extract_person(text) for text in dataset]
    for person in asyncio.as_completed(tasks_get_persons):
        all_persons.append(await person) # (1)!
```

1.  We use `await` here to wait for each task to complete before we append it to the list. This is because `as_completed` returns a coroutine object, not the result of the coroutine. We can also use `asyncio.gather` to do this.

This is a great way to handle large datasets. We can start processing the results as they come in. Espcially if we are streaming data back to a client.

However these methods try to do as much as possible as fast as possible. This can be a problem if we're trying to be polite to the server we're making requests to. This is where rate limiting comes in. While there are libraries that can help with this, we'll be using a semaphore to limit the number of concurrent requests we make as a first defense.

4. **Rate-Limited Gather**: Using semaphores to limit concurrency.

```python
sem = asyncio.Semaphore(2)

async def rate_limited_extract_person(text: str, sem: Semaphore) -> Person:
    async with sem: # (1)!
        return await extract_person(text)

async def rate_limited_gather(sem: Semaphore)
    tasks_get_persons = [rate_limited_extract_person(text, sem) for text in dataset]
    resp = await asyncio.gather(*tasks_get_persons)
```

1.  We use a semaphore to limit the number of concurrent requests we make to 2. This is a great way to balance speed with politeness to the server we're making requests to.

2.  **Rate-Limited As Completed**: Using semaphores to limit concurrency.

```python
sem = asyncio.Semaphore(2)

async def rate_limited_extract_person(text: str, sem: Semaphore) -> Person:
    async with sem: # (1)!
        return await extract_person(text)

async def rate_limited_as_completed(sem: Semaphore)
    all_persons = []
    tasks_get_persons = [rate_limited_extract_person(text, sem) for text in dataset]
    for person in asyncio.as_completed(tasks_get_persons):
        all_persons.append(await person) # (2)!
```

1.  We use a semaphore to limit the number of concurrent requests we make to 2. This is a great way to balance speed with politeness to the server we're making requests to.

2.  We use `await` here to wait for each task to complete before we append it to the list. This is because `as_completed` returns a coroutine object, not the result of the coroutine. We can also use `asyncio.gather` to do this.

Now that we've seen the code, lets look at the results, of processing 7 texts. You can imagine as prompts get longer, or we use gpt-4, the difference between the methods will become more pronounced.

## Analysis of Results

| Method               | Execution Time | Rate Limited (Semaphore) |
| -------------------- | -------------- | ------------------------ |
| For Loop             | 6.17 seconds   |                          |
| Asyncio.gather       | 1.11 seconds   |                          |
| Asyncio.as_completed | 0.87 seconds   |                          |
| Asyncio.gather       | 3.04 seconds   | 2                        |
| Asyncio.as_completed | 3.26 seconds   | 2                        |

## Practical implications for batch processing

Choosing the right approach depends on the task's nature and the desired balance between speed and resource utilization.

### Best Practices for AsyncIO and Batch Processing

#### Tips for Optimal Performance

- Use `asyncio.gather` for speed when handling multiple independent tasks.
- Apply `asyncio.as_completed` for large datasets to process tasks as they complete.
- Implement rate-limiting to avoid overwhelming servers or API endpoints.

#### When to Use Different AsyncIO Strategies

Selecting an AsyncIO strategy should be based on the specific requirements of your task, such as speed, resource constraints, and the nature of the tasks.

## Conclusion

AsyncIO offers a flexible and efficient approach to handling asynchronous tasks in Python, with various strategies to suit different scenarios.

## Potential Use Cases for Instructor and AsyncIO in Python Projects

This approach is particularly beneficial for web scraping, API interactions, and any I/O-bound operations requiring concurrent processing.

This blog post demonstrates the practicality and adaptability of AsyncIO in Python, providing insights and guidance for leveraging its full potential in various programming tasks.

If you enjoy the content or want to try out `Instructor` please check out the [github](https://github.com/jxnl/instructor) and give us a star!
