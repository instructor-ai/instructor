import functools
import inspect
import instructor
import diskcache

from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel

client = instructor.patch(OpenAI())
aclient = instructor.patch(AsyncOpenAI())


class UserDetail(BaseModel):
    name: str
    age: int


cache = diskcache.Cache("./my_cache_directory")


def instructor_cache(func):
    """Cache a function that returns a Pydantic model"""
    return_type = inspect.signature(func).return_annotation
    if not issubclass(return_type, BaseModel):
        raise ValueError("The return type must be a Pydantic model")

    is_async = inspect.iscoroutinefunction(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = f"{func.__name__}-{functools._make_key(args, kwargs, typed=False)}"
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

    @functools.wraps(func)
    async def awrapper(*args, **kwargs):
        key = f"{func.__name__}-{functools._make_key(args, kwargs, typed=False)}"
        # Check if the result is already cached
        if (cached := cache.get(key)) is not None:
            # Deserialize from JSON based on the return type
            if issubclass(return_type, BaseModel):
                return return_type.model_validate_json(cached)

        # Call the function and cache its result
        result = await func(*args, **kwargs)
        serialized_result = result.model_dump_json()
        cache.set(key, serialized_result)

        return result

    return wrapper if not is_async else awrapper


@instructor_cache
def extract(data) -> UserDetail:
    return client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=UserDetail,
        messages=[
            {"role": "user", "content": data},
        ],
    )  # type: ignore


@instructor_cache
async def aextract(data) -> UserDetail:
    return await aclient.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=UserDetail,
        messages=[
            {"role": "user", "content": data},
        ],
    )  # type: ignore


def test_extract():
    import time

    start = time.perf_counter()
    model = extract("Extract jason is 25 years old")
    assert model.name.lower() == "jason"
    assert model.age == 25
    print(f"Time taken: {time.perf_counter() - start}")

    start = time.perf_counter()
    model = extract("Extract jason is 25 years old")
    assert model.name.lower() == "jason"
    assert model.age == 25
    print(f"Time taken: {time.perf_counter() - start}")


async def atest_extract():
    import time

    start = time.perf_counter()
    model = await aextract("Extract jason is 25 years old")
    assert model.name.lower() == "jason"
    assert model.age == 25
    print(f"Time taken: {time.perf_counter() - start}")

    start = time.perf_counter()
    model = await aextract("Extract jason is 25 years old")
    assert model.name.lower() == "jason"
    assert model.age == 25
    print(f"Time taken: {time.perf_counter() - start}")


if __name__ == "__main__":
    test_extract()
    # Time taken: 0.7285366660216823
    # Time taken: 9.841693099588156e-05

    import asyncio

    asyncio.run(atest_extract())
