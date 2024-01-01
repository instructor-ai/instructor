import redis
import functools
import inspect
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


if __name__ == "__main__":
    test_extract()
    # Time taken: 0.798335583996959
    # Time taken: 0.00017016706988215446
