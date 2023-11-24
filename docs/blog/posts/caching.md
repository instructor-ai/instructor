---
draft: False
date: 2023-11-24
slug: python-caching
tags:
  - caching
  - functools
  - python
authors:
  - jxnl
---

# Mastering Caching in Python

> Instructor make working with language models easy, but they are still computationally expensive.

Today, we're diving into optimizing instructor code while maintaining the excellent DX offered by Pydantic models. We'll tackle the challenges of caching Pydantic models, typically incompatible with `pickle`, and explore solutions that use `decorators` like using `functools.cache`. Then, we'll craft custom decorators with `diskcache` and `redis`.

Lets first consider a simple example, using the `OpenAI` Python client to extract user details.

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel

# Enables `response_model`
client = instructor.patch(OpenAI())

class UserDetail(BaseModel):
    name: str
    age: int

def extract(data):
    return client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=UserDetail,
    messages=[
        {"role": "user", "content": data},
    ]
)
```

Now imagine wanting to batch process data, run tests or experiments, or simply call `extract` multiple times. We'll quickly run into performance issues, as the function will be called repeatedly, and the same data will be processed over and over again, costing us time and money.

## 1. `functools.cache` for Simple In-Memory Caching

**When to Use**: Ideal for functions with immutable arguments, called repeatedly with the same parameters in small to medium-sized applications. This makes sense when we might be reusing the same data within a single session. or in an application where we don't need to persist the cache between sessions.

```python
import functools

@functools.cache
def extract(data):
    return client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=UserDetail,
    messages=[
        {"role": "user", "content": data},
    ]
)
```

!!! warning "Changing the Model does not Invalidate the Cache"

    Note that changing the model does not invalidate the cache. This is because the cache key is based on the function's name and arguments, not the model. This means that if we change the model, the cache will still return the old result.

Now we can call `extract` multiple times with the same argument, and the result will be cached in memory for faster access.

```python
import time

start = time.perf_counter()
model = extract("Extract jason is 25 years old")
print(f"Time taken: {time.perf_counter() - start}")

start = time.perf_counter()
model = extract("Extract jason is 25 years old")
print(f"Time taken: {time.perf_counter() - start}")

>>> Time taken: 0.9267581660533324
>>> Time taken: 1.2080417945981026e-06
```

**Benefits**: Easy to implement, provides fast access due to in-memory storage, and requires no additional libraries.

??? question "What is a decorator?"

    A decorator is a function that takes another function and extends the behavior of the latter function without explicitly modifying it. In Python, decorators are functions that take a function as an argument and return a closure.

    ```python
    def decorator(func):
        def wrapper(*args, **kwargs):
            print("Do something before")
            result = func(*args, **kwargs)
            print("Do something after")
            return result
        return wrapper

    @decorator
    def say_hello():
        print("Hello!")

    say_hello()
    >>> "Do something before"
    >>> "Hello!"
    >>> "Do something after"
    ```

## 2. `diskcache` for Persistent, Large Data Caching

**When to Use**: Suitable for applications needing cache persistence between sessions or dealing with large datasets. This is useful when we want to reuse the same data across multiple sessions, or when we need to store large amounts of data!

```python
import functools
import inspect
import instructor
from openai import OpenAI
from pydantic import BaseModel
import diskcache

client = instructor.patch(OpenAI())

class UserDetail(BaseModel):
    name: str
    age: int

cache = diskcache.Cache('./my_cache_directory')

def instructor_cache(func):
    """Cache a function that returns a Pydantic model"""
    return_type = inspect.signature(func).return_annotation
    if not issubclass(return_type, BaseModel):
        raise ValueError("The return type must be a Pydantic model")

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = f"{func.__name__}-{str(args)}-{str(kwargs)}"
        # Check if the result is already cached
        if (cached := cache.get(key)) is not None:
            # Deserialize from JSON based on the return type
            if issubclass(return_type, BaseModel):
                return return_type.model_validate_json(cached)

        # Call the function and cache its result
        result = func(*args, **kwargs)
        serialized_result = result.model_dump_json()
        cache.set(key, serialized_result)

        return result

    return wrapper

@instructor_cache
def extract(data) -> UserDetail:
    return client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=UserDetail,
        messages=[
            {"role": "user", "content": data},
        ]
    )
```

**Benefits**: Reduces computation time for heavy data processing, provides disk-based caching for persistence.

## 2. Redis Caching Decorator for Distributed Systems

**When to Use**: Recommended for distributed systems where multiple processes need to access the cached data, or for applications requiring fast read/write access and handling complex data structures.

```python
import redis
import functools
import inspect
import json
import instructor

from pydantic import BaseModel
from openai import OpenAI

client = instructor.patch(OpenAI())
cache = redis.Redis("localhost")

def instructor_cache(func):
    """Cache a function that returns a Pydantic model"""
    return_type = inspect.signature(func).return_annotation
    if not issubclass(return_type, BaseModel):
        raise ValueError("The return type must be a Pydantic model")

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = f"{func.__name__}-{str(args)}-{str(kwargs)}"
        # Check if the result is already cached
        if (cached := cache.get(key)) is not None:
            # Deserialize from JSON based on the return type
            if issubclass(return_type, BaseModel):
                return return_type.model_validate_json(cached)

        # Call the function and cache its result
        result = func(*args, **kwargs)
        serialized_result = result.model_dump_json()
        cache.set(key, serialized_result)

        return result

    return wrapper


class UserDetail(BaseModel):
    name: str
    age: int

@instructor_cache
def extract(data) -> UserDetail:
    # Assuming client.chat.completions.create returns a UserDetail instance
    return client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=UserDetail,
        messages=[
            {"role": "user", "content": data},
        ]
    )
```

**Benefits**: Scalable for large-scale systems, supports fast in-memory data storage and retrieval, and is versatile for various data types.

!!! note "Looking carefully"

    If you look carefully at the code above you'll notice that we're using the same `instructor_cache` decorator as before. The implemntations is the same, but we're using a different caching backend!

## Conclusion

Choosing the right caching strategy depends on your application's specific needs, such as the size and type of data, the need for persistence, and the system's architecture. Whether it's optimizing a function's performance in a small application or managing large datasets in a distributed environment, Python offers robust solutions to improve efficiency and reduce computational overhead.

If you like the content check out our [GitHub](https://github.com/jxnl/instructor) as give us a star and checkout the library.
