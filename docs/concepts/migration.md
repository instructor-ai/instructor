---
title: Migrating to from_provider
description: Step-by-step guide to migrating from older Instructor patterns to the modern from_provider approach.
---

# Migration Guide

This guide helps you migrate from older Instructor initialization patterns to the modern `from_provider` approach. The `from_provider` function provides a unified interface that works across all providers and simplifies your code.

## Why Migrate?

Migrating to `from_provider` offers several benefits:

- **Simpler code**: Less boilerplate, easier to read
- **Consistent interface**: Same pattern works for all providers
- **Better type safety**: Improved IDE support and type inference
- **Easier provider switching**: Change providers with a single string
- **Future-proof**: This is the recommended pattern going forward

## Migration Patterns

### Pattern 1: Replacing `patch()`

**Before:**

```python
import openai
import instructor

client = openai.OpenAI()
patched_client = instructor.patch(client)

user = patched_client.chat.completions.create(
    response_model=User,
    messages=[{"role": "user", "content": "Extract: John is 30"}],
)
```

**After:**

```python
import instructor

client = instructor.from_provider("openai/gpt-4o-mini")

user = client.create(
    response_model=User,
    messages=[{"role": "user", "content": "Extract: John is 30"}],
)
```

**Changes:**
- Remove `import openai` and `instructor.patch()`
- Use `instructor.from_provider("openai/model-name")`
- Use `client.create()` instead of `client.chat.completions.create()`

### Pattern 2: Replacing `from_openai()`

**Before:**

```python
import openai
from instructor import from_openai

client = openai.OpenAI()
instructor_client = from_openai(client)

user = instructor_client.chat.completions.create(
    response_model=User,
    messages=[{"role": "user", "content": "Extract: John is 30"}],
)
```

**After:**

```python
import instructor

client = instructor.from_provider("openai/gpt-4o-mini")

user = client.create(
    response_model=User,
    messages=[{"role": "user", "content": "Extract: John is 30"}],
)
```

**Changes:**
- Remove `import openai` and `from_openai`
- Use `instructor.from_provider("openai/model-name")`
- Use `client.create()` instead of `client.chat.completions.create()`

### Pattern 3: Replacing `from_anthropic()`

**Before:**

```python
import anthropic
from instructor import from_anthropic

client = anthropic.Anthropic()
instructor_client = from_anthropic(client)

user = instructor_client.messages.create(
    response_model=User,
    messages=[{"role": "user", "content": "Extract: John is 30"}],
)
```

**After:**

```python
import instructor

client = instructor.from_provider("anthropic/claude-3-5-sonnet")

user = client.create(
    response_model=User,
    messages=[{"role": "user", "content": "Extract: John is 30"}],
)
```

**Changes:**
- Remove `import anthropic` and `from_anthropic`
- Use `instructor.from_provider("anthropic/model-name")`
- Use `client.create()` instead of `client.messages.create()`

### Pattern 4: Replacing `from_google()` or `from_genai()`

**Before:**

```python
import google.genai as genai
from instructor import from_genai

client = genai.Client(api_key="your-key")
instructor_client = from_genai(client, model="gemini-pro")

user = instructor_client.generate_content(
    response_model=User,
    contents="Extract: John is 30",
)
```

**After:**

```python
import instructor

client = instructor.from_provider("google/gemini-pro")

user = client.create(
    response_model=User,
    messages=[{"role": "user", "content": "Extract: John is 30"}],
)
```

**Changes:**
- Remove `import google.genai` and `from_genai`
- Use `instructor.from_provider("google/model-name")`
- Use `client.create()` with `messages` parameter

### Pattern 5: Async Clients

**Before:**

```python
import openai
import instructor

client = openai.AsyncOpenAI()
patched_client = instructor.apatch(client)

user = await patched_client.chat.completions.create(
    response_model=User,
    messages=[{"role": "user", "content": "Extract: John is 30"}],
)
```

**After:**

```python
import instructor

client = instructor.from_provider("openai/gpt-4o-mini", async_client=True)

user = await client.create(
    response_model=User,
    messages=[{"role": "user", "content": "Extract: John is 30"}],
)
```

**Changes:**
- Use `async_client=True` parameter
- Use `client.create()` instead of `client.chat.completions.create()`

### Pattern 6: Custom Mode Configuration

**Before:**

```python
import openai
import instructor

client = openai.OpenAI()
patched_client = instructor.patch(client, mode=instructor.Mode.JSON)

user = patched_client.chat.completions.create(
    response_model=User,
    messages=[{"role": "user", "content": "Extract: John is 30"}],
)
```

**After:**

```python
import instructor

client = instructor.from_provider(
    "openai/gpt-4o-mini",
    mode=instructor.Mode.JSON
)

user = client.create(
    response_model=User,
    messages=[{"role": "user", "content": "Extract: John is 30"}],
)
```

**Changes:**
- Pass `mode` parameter to `from_provider`
- Same mode options work as before

## Step-by-Step Migration Checklist

Follow these steps to migrate your code:

### Step 1: Identify Current Pattern

Check which pattern you're using:
- [ ] `instructor.patch(client)`
- [ ] `instructor.apatch(client)`
- [ ] `from_openai(client)`
- [ ] `from_anthropic(client)`
- [ ] `from_genai(client)`
- [ ] Other provider-specific functions

### Step 2: Find Model Name

Identify the model you're using:
- OpenAI: Check `client.model` or look for model names like `"gpt-4o"`, `"gpt-4-turbo"`
- Anthropic: Look for `"claude-3-5-sonnet"`, `"claude-3-opus"`
- Google: Look for `"gemini-pro"`, `"gemini-2.5-flash"`

### Step 3: Replace Client Creation

Replace the old pattern with `from_provider`:

```python
# Old
client = instructor.patch(openai.OpenAI())

# New
client = instructor.from_provider("openai/gpt-4o-mini")
```

### Step 4: Update Method Calls

Change from provider-specific methods to `client.create()`:

```python
# Old (OpenAI)
response = client.chat.completions.create(...)

# Old (Anthropic)
response = client.messages.create(...)

# New (all providers)
response = client.create(...)
```

### Step 5: Update Message Format

Ensure messages use the standard format:

```python
messages = [
    {"role": "user", "content": "Your prompt here"}
]
```

### Step 6: Test Your Code

Run your code and verify it works:
- Check that responses are correct
- Verify error handling still works
- Test async code if applicable

## Common Migration Scenarios

### Scenario 1: Multiple Providers

**Before:**

```python
import openai
import anthropic
import instructor

openai_client = instructor.patch(openai.OpenAI())
anthropic_client = instructor.patch(anthropic.Anthropic())

# Use different clients for different tasks
```

**After:**

```python
import instructor

openai_client = instructor.from_provider("openai/gpt-4o-mini")
anthropic_client = instructor.from_provider("anthropic/claude-3-5-sonnet")

# Same interface, easier to manage
```

### Scenario 2: Environment-Based Provider Selection

**Before:**

```python
import os
import openai
import anthropic
import instructor

provider = os.getenv("LLM_PROVIDER", "openai")

if provider == "openai":
    client = instructor.patch(openai.OpenAI())
elif provider == "anthropic":
    client = instructor.patch(anthropic.Anthropic())
```

**After:**

```python
import os
import instructor

provider = os.getenv("LLM_PROVIDER", "openai")
model_map = {
    "openai": "openai/gpt-4o-mini",
    "anthropic": "anthropic/claude-3-5-sonnet",
}

client = instructor.from_provider(model_map[provider])
```

### Scenario 3: Custom Configuration

**Before:**

```python
import openai
import instructor

client = openai.OpenAI(
    api_key="custom-key",
    organization="org-id",
    timeout=30.0
)
patched_client = instructor.patch(client)
```

**After:**

```python
import instructor

client = instructor.from_provider(
    "openai/gpt-4o-mini",
    api_key="custom-key",
    organization="org-id",
    timeout=30.0
)
```

## Backward Compatibility

The old patterns still work, but they're deprecated:

- `instructor.patch()` - Still works, but use `from_provider` instead
- `instructor.apatch()` - Still works, but use `from_provider` with `async_client=True`
- `from_openai()`, `from_anthropic()`, etc. - Still work, but use `from_provider` instead

You can migrate gradually - both patterns work side by side.

## Troubleshooting Migration Issues

### Issue: Method Not Found

If you see errors like `'Instructor' object has no attribute 'chat'`:

**Problem:** You're using the old method call pattern.

**Solution:** Change from `client.chat.completions.create()` to `client.create()`.

### Issue: Wrong Message Format

If you see errors about message format:

**Problem:** Provider-specific message formats differ.

**Solution:** Use the standard `messages` list format for all providers:

```python
messages = [
    {"role": "user", "content": "Your prompt"}
]
```

### Issue: Model Not Found

If you see errors about invalid model strings:

**Problem:** Model name format is incorrect.

**Solution:** Use format `"provider/model-name"`:

```python
# Correct
"openai/gpt-4o-mini"

# Incorrect
"gpt-4o-mini"  # Missing provider
```

## Migration Examples

### Complete Example: Before and After

**Before:**

```python
import openai
import instructor
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

# Create client
client = openai.OpenAI()
patched_client = instructor.patch(client)

# Extract data
response = patched_client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=User,
    messages=[{"role": "user", "content": "Extract: John is 30"}],
)

print(response.name, response.age)
```

**After:**

```python
import instructor
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

# Create client
client = instructor.from_provider("openai/gpt-4o-mini")

# Extract data
response = client.create(
    response_model=User,
    messages=[{"role": "user", "content": "Extract: John is 30"}],
)

print(response.name, response.age)
```

## Benefits After Migration

After migrating, you'll notice:

1. **Less code**: Fewer imports and lines
2. **Easier testing**: Mock `from_provider` easily
3. **Better IDE support**: Improved autocomplete and type hints
4. **Easier maintenance**: One pattern to maintain
5. **Future features**: Access to new features first

## Related Documentation

- [from_provider Guide](./from_provider.md) - Complete guide to using from_provider
- [from_provider Guide](./from_provider.md) - Complete client configuration guide
- [Patching](./patching.md) - How Instructor enhances clients

