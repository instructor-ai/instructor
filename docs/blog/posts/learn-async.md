---
draft: False
date: 2023-11-13
slug: learn-async
tags:
  - python
  - batch
  - asyncio
  - async
  - async/await
authors:
  - jxnl
---

# Introduction to Batch Processing using `asyncio` and `Instructor`

Today, I will introduce you to various approaches for using asyncio in Python. We will apply this to batch process data using `instructor` and learn how to use `asyncio.gather` and `asyncio.as_completed` for concurrent data processing. Additionally, we will explore how to limit the number of concurrent requests to a server using `asyncio.Semaphore`.

<!-- more -->

!!! notes "Github Example"

    If you want to run the code examples in this article, you can find them on [jxnl/instructor](https://github.com/jxnl/instructor/blob/main/examples/learn-async/run.py)

We will start by defining an `async` function that calls `openai` to extract data, and then examine four different ways to execute it. We will discuss the pros and cons of each approach and analyze the results of running them on a small batch.

## Understanding `asyncio`

`asyncio` is a Python library that enables writing concurrent code using the async/await syntax. It is particularly useful for IO-bound and structured network code. If you are familiar with OpenAI's SDK, you might have encountered two classes: `OpenAI()` and `AsyncOpenAI()`. Today, we will be using the `AsyncOpenAI()` class, which processes data asynchronously.

By utilizing these tools in web applications or batch processing, we can significantly improve performance by handling multiple requests concurrently instead of sequentially.

### Understanding `async` and `await`

We will be using the `async` and `await` keywords to define asynchronous functions. The `async` keyword is used to define a function that returns a coroutine object. The `await` keyword is used to wait for the result of a coroutine object.

If you want to understand the deeper details of `asyncio`, I recommend reading [this article](https://realpython.com/async-io-python/) by Real Python.

### Understanding `gather` vs `as_completed`

In this post we'll show two ways to run tasks concurrently: `asyncio.gather` and `asyncio.as_completed`. The `gather` method is used to run multiple tasks concurrently and return the results as a `list`. The `as_completed` returns a `iterable` is used to run multiple tasks concurrently and return the results as they complete. Another great resource on the differences between the two can be found [here](https://medium.com/dev-bits/a-minimalistic-guide-for-understanding-asyncio-in-python-52c436c244ea).

## Example: Batch Processing

In this example, we will demonstrate how to use `asyncio` for batch processing tasks, specifically for extracting and processing data concurrently. The script will extract data from a list of texts and process it concurrently using `asyncio`.

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

Notice that now there are `async` and `await` keywords in the function definition. This is because we're using the `asyncio` library to run the function concurrently. Now let's define a batch of texts to process.

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

### **`for loop`**: Running tasks sequentially.

```python hl_lines="3"
persons = []
for text in dataset:
    person = await extract_person(text)
    persons.append(person)
```

Even though there is an `await` keyword, we still have to wait for each task to finish before starting the next one. This is because we're using a `for` loop to iterate over the dataset. This method, which uses a `for` loop, will be the slowest among the four methods discussed today.

### **`asyncio.gather`**: Running tasks concurrently.

```python hl_lines="3"
async def gather():
    tasks_get_persons = [extract_person(text) for text in dataset]
    all_persons = await asyncio.gather(*tasks_get_persons) # (1)!
```

1. We use `await` here to wait for all the tasks to finish before assigning the result to `all_persons`. This is because `asyncio.gather` returns a coroutine object, not the result of the coroutine. Alternatively, we can use `asyncio.as_completed` to achieve the same result.

Using `asyncio.gather` allows us to return all the results at once. It is an effective way to speed up our code, but it's not the only way. Particularly, if we have a large dataset, we might not want to wait for everything to finish before starting to process the results. This is where `asyncio.as_completed` comes into play.

### **`asyncio.as_completed`**: Handling tasks as they complete.

```python hl_lines="5 4"
async def as_completed():
    all_persons = []
    tasks_get_persons = [extract_person(text) for text in dataset]
    for person in asyncio.as_completed(tasks_get_persons):
        all_persons.append(await person) # (1)!
```

1. We use `await` here to wait for each task to complete before appending it to the list. This is because `as_completed` returns a coroutine object, not the result of the coroutine. Alternatively, we can use `asyncio.gather` to achieve the same result.

This method is a great way to handle large datasets. We can start processing the results as they come in, especially if we are streaming data back to a client.

However, these methods aim to complete as many tasks as possible as quickly as possible. This can be problematic if we want to be considerate to the server we're making requests to. This is where rate limiting comes into play. While there are libraries available to assist with rate limiting, for our initial defense, we will use a semaphore to limit the number of concurrent requests we make.

!!! note "Ordering of results"

    It is important to note that the order of the results will not be the same as the order of the dataset. This is because the tasks are completed in the order they finish, not the order they were started. If you need to preserve the order of the results, you can use `asyncio.gather` instead.

### **Rate-Limited Gather**: Using semaphores to limit concurrency.

```python hl_lines="4 8 9"
sem = asyncio.Semaphore(2)

async def rate_limited_extract_person(text: str, sem: Semaphore) -> Person:
    async with sem: # (1)!
        return await extract_person(text)

async def rate_limited_gather(sem: Semaphore):
    tasks_get_persons = [rate_limited_extract_person(text, sem) for text in dataset]
    resp = await asyncio.gather(*tasks_get_persons)
```

1. We use a semaphore to limit the number of concurrent requests to 2. This approach strikes a balance between speed and being considerate to the server we're making requests to.

### **Rate-Limited As Completed**: Using semaphores to limit concurrency.

```python hl_lines="4 9 10"
sem = asyncio.Semaphore(2)

async def rate_limited_extract_person(text: str, sem: Semaphore) -> Person:
    async with sem: # (1)!
        return await extract_person(text)

async def rate_limited_as_completed(sem: Semaphore):
    all_persons = []
    tasks_get_persons = [rate_limited_extract_person(text, sem) for text in dataset]
    for person in asyncio.as_completed(tasks_get_persons):
        all_persons.append(await person) # (2)!
```

1. We use a semaphore to limit the number of concurrent requests to 2. This approach strikes a balance between speed and being considerate to the server we're making requests to.

2. We use `await` here to wait for each task to complete before appending it to the list. This is because `as_completed` returns a coroutine object, not the result of the coroutine. Alternatively, we can use `asyncio.gather` to achieve the same result.

Now that we have seen the code, let's examine the results of processing 7 texts. As the prompts become longer or if we use GPT-4, the differences between these methods will become more pronounced.

!!! note "Other Options"

    It is important to also note that here we are using a `semaphore` to limit the number of concurrent requests. However, there are other ways to limit concurrency especially since we have rate limit information from the `openai` request. You can imagine using a library like `ratelimit` to limit the number of requests per second. OR catching rate limit exceptions and using `tenacity` to retry the request after a certain amount of time.

    - [tenacity](https://pypi.org/project/tenacity/)
    - [aiolimiter](https://pypi.org/project/aiolimiter/)

## Results

As you can see, the `for` loop is the slowest, while `asyncio.as_completed` and `asyncio.gather` are the fastest without any rate limiting.

| Method               | Execution Time | Rate Limited (Semaphore) |
| -------------------- | -------------- | ------------------------ |
| For Loop             | 6.17 seconds   |                          |
| Asyncio.gather       | 0.85 seconds   |                          |
| Asyncio.as_completed | 0.95 seconds   |                          |
| Asyncio.gather       | 3.04 seconds   | 2                        |
| Asyncio.as_completed | 3.26 seconds   | 2                        |

## Practical implications of batch processing

The choice of approach depends on the task's nature and the desired balance between speed and resource utilization.

Here are some guidelines to consider:

- Use `asyncio.gather` for handling multiple independent tasks quickly.
- Apply `asyncio.as_completed` for large datasets to process tasks as they complete.
- Implement rate-limiting to avoid overwhelming servers or API endpoints.

If you find the content helpful or want to try out `Instructor`, please visit our [GitHub](https://github.com/jxnl/instructor) page and give us a star!
