# OpenAI Starter Example

This directory contains a minimal example showing how to use Instructor with OpenAI.

## Quick Start

```bash
pip install instructor openai
export OPENAI_API_KEY="your-api-key"
python run.py
```

## What This Example Shows

1. **Patching the OpenAI client** with `instructor.from_openai()`
2. **Defining a response model** using Pydantic's `BaseModel`
3. **Extracting structured data** via `response_model` parameter

## Output

```
User(name='John', age=25)
```

## Next Steps

- See [instructor docs](https://python.useinstructor.com/) for advanced usage
- Check other examples in this repository for more complex patterns
