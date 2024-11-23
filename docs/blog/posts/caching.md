---
authors:
- jxnl
categories:
- Performance Optimization
comments: true
date: 2023-11-26
description: Learn caching techniques in Python using Pydantic models with functools,
  diskcache, and Redis for improved performance and efficiency.
draft: false
slug: python-caching
tags:
- Python
- Caching
- Pydantic
- Performance Optimization
- Redis
---

# Introduction to Caching in Python

> Instructor makes working with language models easy, but they are still computationally expensive.

Today, we're diving into optimizing instructor code while maintaining the excellent DX offered by [Pydantic](https://docs.pydantic.dev/latest/) models. We'll tackle the challenges of caching Pydantic models, typically incompatible with `pickle`, and explore solutions that use `decorators` like `functools.cache`. Then, we'll craft custom decorators with `diskcache` and `redis` to support persistent caching and distributed systems.

<!-- more -->

Let's first consider our canonical example, using the `OpenAI` Python client to extract user details.

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel

# Enables `response_model`
client = instructor.from_openai(OpenAI())


class UserDetail(BaseModel):
    name: str
    age: int


def extract(data) -> UserDetail:
    return client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=UserDetail,
        messages=[
            {"role": "user", "content": data},
        ],
    )
```

Now imagine batch processing data, running tests or experiments, or simply calling `extract` multiple times over a workflow. We'll quickly run into performance issues, as the function may be called repeatedly, and the same data will be processed over and over again, costing us time and money.

## 1. `functools.cache` for Simple In-Memory Caching

**When to Use**: Ideal for functions with immutable arguments, called repeatedly with the same parameters in small to medium-sized applications. This makes sense when we might be reusing the same data within a single session or in an application where we don't need to persist the cache between sessions.

```python
import functools


@functools.cache
def extract(data):
    return client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=UserDetail,
        messages=[
            {"role": "user", "content": data},
        ],
    )
```

!!! warning "Changing the Model does not Invalidate the Cache"

    Note that changing the model does not invalidate the cache. This is because the cache key is based on the function's name and arguments, not the model. This means that if we change the model, the cache will still return the old result.

Now we can call `extract` multiple times with the same argument, and the result will be cached in memory for faster access.

```python hl_lines="4 8 12"
import time

start = time.perf_counter()  # (1)
model = extract("Extract jason is 25 years old")
print(f"Time taken: {time.perf_counter() - start}")

start = time.perf_counter()
model = extract("Extract jason is 25 years old")  # (2)
print(f"Time taken: {time.perf_counter() - start}")

#> Time taken: 0.92
#> Time taken: 1.20e-06 # (3)
```

1. Using `time.perf_counter()` to measure the time taken to run the function is better than using `time.time()` because it's more accurate and less susceptible to system clock changes.
2. The second time we call `extract`, the result is returned from the cache, and the function is not called.
3. The second call to `extract` is much faster because the result is returned from the cache!

**Benefits**: Easy to implement, provides fast access due to in-memory storage, and requires no additional libraries.

??? question "What is a decorator?"

    A decorator is a function that takes another function and extends the behavior of the latter function without explicitly modifying it. In Python, decorators are functions that take a function as an argument and return a closure.

    ```python hl_lines="3-5 9"
    def decorator(func):
        def wrapper(*args, **kwargs):
            print("Do something before")  # (1)
            #> Do something before
            result = func(*args, **kwargs)
            print("Do something after")  # (2)
            #> Do something after
            return result

        return wrapper


    @decorator
    def say_hello():
        #> Hello!
        print("Hello!")


    say_hello()
    #> "Do something before"
    #> "Hello!"
    #> "Do something after"
    ```

    1. The code is executed before the function is called
    2. The code is executed after the function is called

## 2. `diskcache` for Persistent, Large Data Caching

??? note "Copy Caching Code"

    We'll be using the same `instructor_cache` decorator for both `diskcache` and `redis` caching. You can copy the code below and use it for both examples.

    ```python
    import functools
    import inspect
    import diskcache

    cache = diskcache.Cache('./my_cache_directory')  # (1)


    def instructor_cache(func):
        """Cache a function that returns a Pydantic model"""
        return_type = inspect.signature(func).return_annotation
        if not issubclass(return_type, BaseModel):  # (2)
            raise ValueError("The return type must be a Pydantic model")

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = f"{func.__name__}-{functools._make_key(args, kwargs, typed=False)}"
            # Check if the result is already cached
            if (cached := cache.get(key)) is not None:
                # Deserialize from JSON based on the return type
                return return_type.model_validate_json(cached)

            # Call the function and cache its result
            result = func(*args, **kwargs)
            serialized_result = result.model_dump_json()
            cache.set(key, serialized_result)

            return result

        return wrapper
    ```

    1. We create a new `diskcache.Cache` instance to store the cached data. This will create a new directory called `my_cache_directory` in the current working directory.
    2. We only want to cache functions that return a Pydantic model to simplify serialization and deserialization logic in this example code

    Remember that you can change this code to support non-Pydantic models, or to use a different caching backend. More over, don't forget that this cache does not invalidate when the model changes, so you might want to encode the `Model.model_json_schema()` as part of the key.

**When to Use**: Suitable for applications needing cache persistence between sessions or dealing with large datasets. This is useful when we want to reuse the same data across multiple sessions, or when we need to store large amounts of data!

```python hl_lines="10"
import functools
import inspect
import instructor
import diskcache

from openai import OpenAI
from pydantic import BaseModel

client = instructor.from_openai(OpenAI())
cache = diskcache.Cache('./my_cache_directory')


def instructor_cache(func):
    """Cache a function that returns a Pydantic model"""
    return_type = inspect.signature(func).return_annotation  # (4)
    if not issubclass(return_type, BaseModel):  # (1)
        raise ValueError("The return type must be a Pydantic model")

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = (
            f"{func.__name__}-{functools._make_key(args, kwargs, typed=False)}"  #  (2)
        )
        # Check if the result is already cached
        if (cached := cache.get(key)) is not None:
            # Deserialize from JSON based on the return type (3)
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
    return client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=UserDetail,
        messages=[
            {"role": "user", "content": data},
        ],
    )
```

1. We only want to cache functions that return a Pydantic model to simplify serialization and deserialization logic
2. We use functool's `_make_key` to generate a unique key based on the function's name and arguments. This is important because we want to cache the result of each function call separately.
3. We use Pydantic's `model_validate_json` to deserialize the cached result into a Pydantic model.
4. We use `inspect.signature` to get the function's return type annotation, which we use to validate the cached result.

**Benefits**: Reduces computation time for heavy data processing, provides disk-based caching for persistence.

## 2. Redis Caching Decorator for Distributed Systems

??? note "Copy Caching Code"

    We'll be using the same `instructor_cache` decorator for both `diskcache` and `redis` caching. You can copy the code below and use it for both examples.

    ```python
    import functools
    import inspect
    import redis

    cache = redis.Redis("localhost")


    def instructor_cache(func):
        """Cache a function that returns a Pydantic model"""
        return_type = inspect.signature(func).return_annotation
        if not issubclass(return_type, BaseModel):
            raise ValueError("The return type must be a Pydantic model")

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = f"{func.__name__}-{functools._make_key(args, kwargs, typed=False)}"
            # Check if the result is already cached
            if (cached := cache.get(key)) is not None:
                # Deserialize from JSON based on the return type
                return return_type.model_validate_json(cached)

            # Call the function and cache its result
            result = func(*args, **kwargs)
            serialized_result = result.model_dump_json()
            cache.set(key, serialized_result)

            return result

        return wrapper
    ```

    Remember that you can change this code to support non-Pydantic models, or to use a different caching backend. More over, don't forget that this cache does not invalidate when the model changes, so you might want to encode the `Model.model_json_schema()` as part of the key.

**When to Use**: Recommended for distributed systems where multiple processes need to access the cached data, or for applications requiring fast read/write access and handling complex data structures.

```python
import redis
import functools
import inspect
import instructor

from pydantic import BaseModel
from openai import OpenAI

client = instructor.from_openai(OpenAI())
cache = redis.Redis("localhost")


def instructor_cache(func):
    """Cache a function that returns a Pydantic model"""
    return_type = inspect.signature(func).return_annotation
    if not issubclass(return_type, BaseModel):  # (1)
        raise ValueError("The return type must be a Pydantic model")

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = f"{func.__name__}-{functools._make_key(args, kwargs, typed=False)}"  # (2)
        # Check if the result is already cached
        if (cached := cache.get(key)) is not None:
            # Deserialize from JSON based on the return type
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
        ],
    )
```

1. We only want to cache functions that return a Pydantic model to simplify serialization and deserialization logic
2. We use functool's `_make_key` to generate a unique key based on the function's name and arguments. This is important because we want to cache the result of each function call separately.

**Benefits**: Scalable for large-scale systems, supports fast in-memory data storage and retrieval, and is versatile for various data types.

!!! note "Looking carefully"

    If you look carefully at the code above you'll notice that we're using the same `instructor_cache` decorator as before. The implementation is the same, but we're using a different caching backend!

## Conclusion

Choosing the right caching strategy depends on your application's specific needs, such as the size and type of data, the need for persistence, and the system's architecture. Whether it's optimizing a function's performance in a small application or managing large datasets in a distributed environment, Python offers robust solutions to improve efficiency and reduce computational overhead.

If you'd like to use this code, try to send it over to ChatGPT to understand it more, and to add additional features that might matter for you, for example, the cache isn't invalidated when your BaseModel changes, so you might want to encode the `Model.model_json_schema()` as part of the key.

If you like the content check out our [GitHub](https://github.com/jxnl/instructor) as give us a star and checkout the library.