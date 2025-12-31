---
title: "Python Retry Logic with Tenacity and Instructor | Complete Guide"
description: "Master Python retry logic with Tenacity library and Instructor for LLM applications. Learn exponential backoff, conditional retries, rate limit handling, and robust error recovery patterns."
keywords: "tenacity python, python retry, instructor retry logic, exponential backoff, python error handling, LLM retry, API retry, python tenacity library, automatic retries, python resilience, tenacity library python, retry decorator python"
---

# Python Retry Logic with Tenacity and Instructor

_Implement robust retry mechanisms for LLM applications with automatic error handling, exponential backoff, and intelligent failure recovery using Python's Tenacity library._

## What is Tenacity?

**Tenacity** is a powerful Python library that simplifies implementing retry logic in your applications. It provides decorators and utilities for handling transient failures, rate limits, and validation errors with intelligent backoff strategies.

When combined with Instructor, Tenacity creates a robust foundation for LLM applications that can handle API failures, network issues, and validation errors gracefully.

## Why Use Tenacity with Instructor?

- **Automatic Retries**: Handle transient failures without manual intervention
- **Exponential Backoff**: Intelligent delay strategies to avoid overwhelming APIs
- **Conditional Retries**: Retry only on specific error types
- **Rate Limit Handling**: Respect API rate limits with smart delays
- **Validation Error Recovery**: Retry when LLM outputs fail validation
- **Circuit Breaker Patterns**: Prevent cascading failures

## Complete Setup: Basic Retry Implementation

Here's a complete, self-contained example showing how to set up retry logic with Tenacity and Instructor:

```python
import instructor
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import BaseModel, field_validator
from openai import RateLimitError, APIError
import time

# Set up the client with Instructor
client = instructor.from_provider("openai/gpt-4.1-mini")

class UserInfo(BaseModel):
    name: str
    age: int
    email: str

    @field_validator('age')
    @classmethod
    def validate_age(cls, v):
        if v < 0 or v > 150:
            raise ValueError(f"Age {v} is invalid")
        return v

    @field_validator('email')
    @classmethod
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError(f"Invalid email: {v}")
        return v.lower()

# Sample data for testing
test_texts = [
    "John is 30 years old with email john@example.com",
    "Sarah is 25 with email sarah@test.com",
    "Mike is 35 and his email is mike@demo.org"
]
```

## Method 1: Basic Retry with Exponential Backoff

```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def extract_user_info(text: str) -> UserInfo:
    """Extract user information with basic retry logic."""
    return client.create(
        response_model=UserInfo,
        messages=[{"role": "user", "content": f"Extract user info: {text}"}]
    )

# Usage example
try:
    user = extract_user_info("John is 30 years old with email john@example.com")
    print(f"Success: {user.name}, {user.age}, {user.email}")
except Exception as e:
    print(f"Failed after retries: {e}")
```

## Method 2: Conditional Retries for Specific Errors

```python
@retry(
    retry=retry_if_exception_type((RateLimitError, APIError)),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=1, max=60)
)
def robust_extraction(text: str) -> UserInfo:
    """Retry only on specific API errors."""
    return client.create(
        response_model=UserInfo,
        messages=[{"role": "user", "content": text}]
    )

# This will retry on rate limits and API errors, but not validation errors
try:
    user = robust_extraction("Extract: Sarah is 25 with email sarah@test.com")
    print(f"Extracted: {user.name}")
except Exception as e:
    print(f"Failed: {e}")
```

## Method 3: Validation Error Retries

```python
from pydantic import ValidationError

@retry(
    retry=retry_if_exception_type(ValidationError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def extract_with_validation(text: str) -> UserInfo:
    """Retry when Pydantic validation fails."""
    return client.create(
        response_model=UserInfo,
        messages=[{"role": "user", "content": text}]
    )

# This will retry if the LLM returns invalid data
try:
    user = extract_with_validation("Extract: Mike is 35 and his email is mike@demo.org")
    print(f"Validated: {user.name}")
except Exception as e:
    print(f"Validation failed: {e}")
```

## Method 4: Custom Retry Conditions

```python
from tenacity import retry, retry_if_result, stop_after_attempt

def should_retry(result: UserInfo) -> bool:
    """Custom retry logic based on result content."""
    # Retry if age is invalid or email is missing
    return result.age < 0 or result.age > 150 or not result.email

@retry(
    retry=retry_if_result(should_retry),
    stop=stop_after_attempt(3)
)
def extract_valid_user(text: str) -> UserInfo:
    """Retry based on result validation."""
    return client.create(
        response_model=UserInfo,
        messages=[{"role": "user", "content": text}]
    )

# This will retry if the result doesn't meet our quality criteria
try:
    user = extract_valid_user("Extract: Alice is 28 with email alice@example.com")
    print(f"Quality check passed: {user.name}")
except Exception as e:
    print(f"Quality check failed: {e}")
```

## Method 5: Advanced Retry Strategies

### Rate Limit Specific Retry

```python
@retry(
    retry=retry_if_exception_type(RateLimitError),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=1, max=120),
    before_sleep=lambda retry_state: print(f"Rate limited, waiting... (attempt {retry_state.attempt_number})")
)
def rate_limit_safe_extraction(text: str) -> UserInfo:
    """Handle rate limits with longer delays."""
    return client.create(
        response_model=UserInfo,
        messages=[{"role": "user", "content": text}]
    )
```

### Network Error Retry

```python
import requests
from tenacity import retry, retry_if_exception_type, wait_random_exponential

@retry(
    retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
    stop=stop_after_attempt(4),
    wait=wait_random_exponential(multiplier=1, min=4, max=30)
)
def network_resilient_extraction(text: str) -> UserInfo:
    """Handle network issues with random exponential backoff."""
    return client.create(
        response_model=UserInfo,
        messages=[{"role": "user", "content": text}]
    )
```

## Method 6: Context-Based Validation with Retries

The `context` parameter allows you to pass additional data to your validators, enabling dynamic validation rules. This is particularly useful when validation depends on external data or runtime conditions.

```python
import instructor
from pydantic import BaseModel, ValidationInfo, field_validator
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import ValidationError

client = instructor.from_provider("openai/gpt-4.1-mini")

class Citation(BaseModel):
    """A claim with a supporting quote from source text."""
    claim: str
    quote: str
    
    @field_validator('quote')
    @classmethod
    def verify_quote_exists(cls, v: str, info: ValidationInfo):
        """Verify that the quote actually exists in the source text."""
        context = info.context
        if context:
            source_text = context.get('source_text', '')
            if v not in source_text:
                raise ValueError(
                    f"Quote '{v}' not found in source text. "
                    "All quotes must be exact substrings from the source."
                )
        return v

@retry(
    retry=retry_if_exception_type(ValidationError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def extract_citation_with_context(claim: str, source_text: str) -> Citation:
    """Extract a citation and verify it exists in the source text."""
    return client.create(
        response_model=Citation,
        messages=[
            {
                "role": "system",
                "content": "Extract the claim and find an exact quote from the source text that supports it."
            },
            {
                "role": "user",
                "content": "Source: {{ source_text }}\n\nClaim to verify: {{ claim }}"
            }
        ],
        context={"source_text": source_text, "claim": claim}
    )

source = "The Eiffel Tower was completed in 1889 and stands 330 meters tall."
claim = "The Eiffel Tower is over 300 meters tall"

try:
    citation = extract_citation_with_context(claim, source)
    print(f"Claim: {citation.claim}")
    print(f"Supporting quote: {citation.quote}")
except Exception as e:
    print(f"Failed to extract valid citation: {e}")
```

### Context with Templating

You can also combine context-based validation with Jinja templating for even more powerful validation patterns:

```python
import instructor
from pydantic import BaseModel, ValidationInfo, field_validator
from tenacity import retry, stop_after_attempt
from typing import List

client = instructor.from_provider("openai/gpt-4.1-mini")

class ProductReview(BaseModel):
    """A product review with context-aware validation."""
    product_name: str
    rating: int
    comment: str
    
    @field_validator('product_name')
    @classmethod
    def validate_product_exists(cls, v: str, info: ValidationInfo):
        """Ensure the product name matches one of the allowed products."""
        context = info.context
        if context:
            allowed_products = context.get('products', [])
            if v not in allowed_products:
                raise ValueError(
                    f"Product '{v}' not recognized. "
                    f"Must be one of: {', '.join(allowed_products)}"
                )
        return v

@retry(stop=stop_after_attempt(3))
def extract_review_with_validation(
    review_text: str,
    allowed_products: List[str]
) -> ProductReview:
    """Extract a review ensuring the product is valid."""
    return client.create(
        response_model=ProductReview,
        messages=[
            {
                "role": "system",
                "content": "Extract product review details. Valid products: {{ products|join(', ') }}"
            },
            {
                "role": "user", 
                "content": "Review: {{ review }}"
            }
        ],
        context={
            "products": allowed_products,
            "review": review_text
        }
    )

allowed = ["Widget Pro", "Gadget Max", "Device Ultra"]
review_text = "The Widget Pro is amazing! Rating: 5 stars. Highly recommended."

try:
    review = extract_review_with_validation(review_text, allowed)
    print(f"Product: {review.product_name}, Rating: {review.rating}/5")
except Exception as e:
    print(f"Validation failed: {e}")
```

### Key Benefits of Context-Based Validation

1. **Dynamic Rules**: Validation rules can adapt based on runtime data
2. **External Data**: Validate against databases, APIs, or other external sources
3. **Multi-Field Dependencies**: Use context to validate relationships between fields
4. **Reusable Validators**: Same validator can work with different context data

For more details on context and templating, see the [Templating Guide](./templating.md) and [Validation Context Example](../examples/exact_citations.md).

## Method 6: Logging and Monitoring

```python
import logging
from tenacity import before_log, after_log

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    before=before_log(logger, logging.INFO),
    after=after_log(logger, logging.ERROR)
)
def logged_extraction(text: str) -> UserInfo:
    """Extract with comprehensive logging."""
    return client.create(
        response_model=UserInfo,
        messages=[{"role": "user", "content": text}]
    )

# This will log each retry attempt
try:
    user = logged_extraction("Extract: Bob is 32 with email bob@example.com")
    print(f"Logged extraction: {user.name}")
except Exception as e:
    print(f"Logged failure: {e}")
```

## Method 7: Circuit Breaker Pattern

```python
from functools import lru_cache
from tenacity import retry, stop_after_attempt, wait_exponential

@lru_cache(maxsize=1)
def get_client():
    """Cache the client to avoid repeated initialization."""
    return instructor.from_provider("openai/gpt-4.1-mini")

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def circuit_breaker_extraction(text: str) -> UserInfo:
    """Extract with circuit breaker pattern."""
    client = get_client()
    return client.create(
        response_model=UserInfo,
        messages=[{"role": "user", "content": text}]
    )
```

## Method 8: Batch Processing with Retries

```python
import asyncio
from tenacity import retry, stop_after_attempt

@retry(stop=stop_after_attempt(3))
async def process_batch(texts: list[str]) -> list[UserInfo]:
    """Process multiple texts with retry logic."""
    client = instructor.from_provider("openai/gpt-4.1-mini")
    results = []

    for text in texts:
        try:
            result = await client.create(
                response_model=UserInfo,
                messages=[{"role": "user", "content": text}]
            )
            results.append(result)
        except Exception as e:
            print(f"Failed to process: {text}, error: {e}")
            continue

    return results

# Async batch processing
async def run_batch_processing():
    results = await process_batch(test_texts)
    print(f"Successfully processed {len(results)} items")

# Run async batch processing
# asyncio.run(run_batch_processing())
```

## Integration with Instructor's Built-in Retries

Instructor has built-in retry support that works alongside Tenacity:

```python
from instructor import Mode

# Instructor's built-in retry with custom settings
client = instructor.from_provider(
    "openai/gpt-4.1-mini",
    mode=Mode.JSON,
    max_retries=3,
    retry_delay=1,
)

# Combine with Tenacity for additional resilience
@retry(stop=stop_after_attempt(2))
def double_retry_extraction(text: str) -> UserInfo:
    """Combine Instructor and Tenacity retries."""
    return client.create(
        response_model=UserInfo,
        messages=[{"role": "user", "content": text}]
    )
```

## Failed Attempts Tracking

Instructor's retry system now tracks all failed attempts with detailed context for better debugging and error handling.

### Enhanced Error Context

When retries fail, exceptions include comprehensive failure history:

```python
import instructor
from instructor.core.exceptions import InstructorRetryException
from pydantic import BaseModel, field_validator

client = instructor.from_provider("openai/gpt-4.1-mini")

class UserInfo(BaseModel):
    name: str
    age: int
    
    @field_validator('age')
    @classmethod
    def validate_age(cls, v):
        if v < 0 or v > 150:
            raise ValueError(f"Age {v} is invalid")
        return v

try:
    result = client.create(
        response_model=UserInfo,
        messages=[{"role": "user", "content": "Extract: John is -5 years old"}],
        max_retries=3
    )
except InstructorRetryException as e:
    # Access failed attempts for debugging
    print(f"Failed after {e.n_attempts} attempts")
    for attempt in e.failed_attempts:
        print(f"Attempt {attempt.attempt_number}: {attempt.exception}")
    
    # Exception string includes rich context:
    # <failed_attempts>
    #   <generation number="1">
    #     <exception>ValidationError: Age -5 is invalid</exception>
    #     <completion>{"name": "John", "age": -5}</completion>
    #   </generation>
    # </failed_attempts>
```

### Improved Reask Behavior

Failed attempts are automatically propagated to reask handlers, enabling:
- **Contextual error messages** - LLMs receive previous failure information
- **Progressive corrections** - Each retry learns from past mistakes  
- **Smarter retry strategies** - Better pattern recognition across attempts

## Best Practices for Tenacity with Instructor

### 1. Choose Appropriate Retry Strategies

```python
# For rate limits - longer delays
@retry(
    wait=wait_exponential(multiplier=2, min=1, max=60),
    stop=stop_after_attempt(5)
)

# For validation errors - shorter delays
@retry(
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(3)
)

# For network issues - moderate delays
@retry(
    wait=wait_exponential(multiplier=1.5, min=2, max=30),
    stop=stop_after_attempt(4)
)
```

### 2. Error-Specific Retry Logic

```python
from tenacity import retry, retry_if_exception_type, stop_after_attempt

# Different strategies for different errors
@retry(
    retry=retry_if_exception_type(RateLimitError),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=1, max=120)
)
def handle_rate_limits(text: str) -> UserInfo:
    return client.create(
        response_model=UserInfo,
        messages=[{"role": "user", "content": text}]
    )

@retry(
    retry=retry_if_exception_type(ValidationError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10)
)
def handle_validation_errors(text: str) -> UserInfo:
    return client.create(
        response_model=UserInfo,
        messages=[{"role": "user", "content": text}]
    )
```

### 3. Performance Monitoring

```python
import time
from tenacity import retry, stop_after_attempt

@retry(stop=stop_after_attempt(3))
def monitored_extraction(text: str) -> UserInfo:
    """Extract with performance monitoring."""
    start_time = time.time()

    try:
        result = client.create(
            response_model=UserInfo,
            messages=[{"role": "user", "content": text}]
        )

        end_time = time.time()
        print(f"Extraction took {end_time - start_time:.2f} seconds")
        return result

    except Exception as e:
        end_time = time.time()
        print(f"Extraction failed after {end_time - start_time:.2f} seconds: {e}")
        raise
```

## Common Use Cases and Patterns

### 1. API Rate Limit Handling

```python
@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=2, min=1, max=120),
    stop=stop_after_attempt(5)
)
def api_friendly_extraction(text: str) -> UserInfo:
    """Respect API rate limits with exponential backoff."""
    return client.create(
        response_model=UserInfo,
        messages=[{"role": "user", "content": text}]
    )
```

### 2. Validation Error Recovery

```python
@retry(
    retry=retry_if_exception_type(ValidationError),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(3)
)
def validation_resilient_extraction(text: str) -> UserInfo:
    """Recover from validation errors with retries."""
    return client.create(
        response_model=UserInfo,
        messages=[{"role": "user", "content": text}]
    )
```

### 3. Network Resilience

```python
@retry(
    retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(4)
)
def network_resilient_extraction(text: str) -> UserInfo:
    """Handle network issues gracefully."""
    return client.create(
        response_model=UserInfo,
        messages=[{"role": "user", "content": text}]
    )
```

## Troubleshooting Common Issues

### 1. Infinite Retries

**Problem**: Function keeps retrying indefinitely
**Solution**: Always set appropriate `stop` conditions

```python
# Good: Set maximum attempts
@retry(stop=stop_after_attempt(3))

# Bad: No stop condition (could retry forever)
@retry()  # Don't do this!
```

### 2. Too Many Retries

**Problem**: Too many retry attempts causing delays
**Solution**: Adjust retry strategy based on error type

```python
# For validation errors - fewer retries
@retry(
    retry=retry_if_exception_type(ValidationError),
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=5)
)

# For rate limits - more retries with longer delays
@retry(
    retry=retry_if_exception_type(RateLimitError),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=1, max=60)
)
```

### 3. Rate Limit Issues

**Problem**: Still hitting rate limits despite retries
**Solution**: Use longer delays and respect rate limit headers

```python
@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=2, min=1, max=120),
    stop=stop_after_attempt(5)
)
def rate_limit_respectful_extraction(text: str) -> UserInfo:
    return client.create(
        response_model=UserInfo,
        messages=[{"role": "user", "content": text}]
    )
```

## Debugging Retry Behavior

```python
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, before_log, after_log

logging.basicConfig(level=logging.INFO)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    before=before_log(logging.getLogger(__name__), logging.INFO),
    after=after_log(logging.getLogger(__name__), logging.ERROR)
)
def debug_extraction(text: str) -> UserInfo:
    """Extract with detailed retry logging."""
    return client.create(
        response_model=UserInfo,
        messages=[{"role": "user", "content": text}]
    )

# This will show detailed retry information
try:
    user = debug_extraction("Extract: Debug user info")
    print(f"Debug extraction successful: {user.name}")
except Exception as e:
    print(f"Debug extraction failed: {e}")
```

## Performance Considerations

- **Retry Overhead**: Each retry adds latency, so choose strategies carefully
- **Cost Impact**: More retries = higher API costs
- **User Experience**: Balance reliability with responsiveness
- **Resource Usage**: Monitor memory and CPU usage during retry storms

## Complete Example: Main Function

```python
def main():
    """Run all retry methods and demonstrate their usage."""
    print("=== Python Tenacity Retry Logic with Instructor ===\n")

    # Test different retry strategies
    strategies = [
        ("Basic Retry", extract_user_info),
        ("Conditional Retry", robust_extraction),
        ("Validation Retry", extract_with_validation),
        ("Custom Retry", extract_valid_user),
        ("Rate Limit Retry", rate_limit_safe_extraction)
    ]

    test_text = "John is 30 years old with email john@example.com"

    for name, strategy in strategies:
        print(f"\n--- Testing {name} ---")
        try:
            start_time = time.time()
            user = strategy(test_text)
            end_time = time.time()
            print(f"✓ Success: {user.name} ({end_time - start_time:.2f}s)")
        except Exception as e:
            print(f"✗ Failed: {e}")

if __name__ == "__main__":
    main()
```

## Key Takeaways

1. **Tenacity** provides powerful retry decorators for Python applications
2. **Exponential backoff** prevents overwhelming APIs during failures
3. **Conditional retries** allow fine-grained control over retry behavior
4. **Error-specific strategies** optimize retry logic for different failure types
5. **Monitoring and logging** help debug and optimize retry behavior

## Related Resources

- [Tenacity Documentation](https://tenacity.readthedocs.io/)
- [Instructor Error Handling](./error_handling.md)
- [Validation Best Practices](./validation.md)
- [Async Processing Guide](../blog/posts/learn-async.md)
- [Python Retry Patterns](https://pypi.org/project/tenacity/)

---

**Next Steps**: Learn about [error handling patterns](./error_handling.md) or explore [async processing](../blog/posts/learn-async.md) for high-performance applications.
