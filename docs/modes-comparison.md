---
title: Mode Comparison Guide
description: Compare different modes available in Instructor and understand when to use each
---

# Instructor Mode Comparison Guide

Instructor supports multiple "modes" for different LLM providers. This guide helps you understand when to use each mode and what their trade-offs are.

## Overview of Available Modes

Instructor's modes determine how structured data is requested from and extracted from LLM responses. The right mode depends on your provider, model, and specific needs.

Here's a quick comparison:

| Mode | Description | Best For | Providers |
|------|-------------|----------|-----------|
| `TOOLS` | Uses function/tool calling API | Most OpenAI use cases | OpenAI, Azure, LiteLLM, etc. |
| `JSON` | Asks for direct JSON output | Models without tool calling | Most providers |
| `FUNCTIONS` (Legacy) | Legacy OpenAI function calling | Backward compatibility | OpenAI, Azure |
| `PARALLEL_TOOLS` | Runs multiple tools in parallel | Complex, multi-step tasks | OpenAI only |
| `MD_JSON` | JSON in Markdown code blocks | Cleaner prompts | Most providers |
| `TOOLS_STRICT` | Stricter version of TOOLS | Production systems | OpenAI, Azure |
| `JSON_O1` | One-shot completion with JSON | Simple extractions | OpenAI, Azure |
| `ANTHROPIC_TOOLS` | Anthropic's tool calling | Claude models | Anthropic |
| `ANTHROPIC_JSON` | Direct JSON with Claude | Claude models | Anthropic |
| `GEMINI_TOOLS` | Google's function calling | Gemini models | Google |
| `GEMINI_JSON` | Direct JSON with Gemini | Gemini models | Google |
| `VERTEXAI_TOOLS` | Vertex AI function calling | Vertex AI models | Vertex AI |
| `VERTEXAI_JSON` | Direct JSON with Vertex AI | Vertex AI models | Vertex AI |
| `COHERE_TOOLS` | Cohere's tool calling | Cohere models | Cohere |

## Detailed Mode Explanations

### OpenAI-Compatible Modes

#### `TOOLS` Mode

```python
client = instructor.from_openai(
    OpenAI(),
    mode=instructor.Mode.TOOLS
)
```

This is the **recommended mode** for OpenAI models. It uses OpenAI's tool calling API to:

- Provide a clear schema to the model
- Ensure proper formatting
- Enable validation and retries

**Best for**: Most OpenAI use cases, especially with GPT-3.5 and GPT-4 models.

**Advantages**:
- Most reliable
- Best for complex structures
- Works with all current OpenAI models

#### `JSON` Mode

```python
client = instructor.from_openai(
    OpenAI(),
    mode=instructor.Mode.JSON
)
```

This mode instructs the model to output direct JSON. It:
- Works with almost any model
- Is slightly less reliable than TOOLS
- May be better for very simple structures

**Best for**: Models that don't support tool calling or simple extractions.

**Advantages**:
- Works with more models
- Can be more token-efficient
- Simpler prompting

#### `MD_JSON` Mode

```python
client = instructor.from_openai(
    OpenAI(),
    mode=instructor.Mode.MD_JSON
)
```

This mode instructs the model to output JSON inside a Markdown code block. It:
- Works with most models
- Can make prompts more readable
- May lead to cleaner outputs

**Best for**: Simple structures when you want cleaner outputs.

### Anthropic Modes

#### `ANTHROPIC_TOOLS` Mode

```python
client = instructor.from_anthropic(
    Anthropic(),
    mode=instructor.Mode.ANTHROPIC_TOOLS
)
```

This mode uses Anthropic's Tool Calling API, available on Claude 3+ models. It:
- Provides structured tool definitions
- Ensures proper response format
- Works well with complex structures

**Best for**: Complex structured outputs with Claude 3+ models.

#### `ANTHROPIC_JSON` Mode

```python
client = instructor.from_anthropic(
    Anthropic(),
    mode=instructor.Mode.ANTHROPIC_JSON
)
```

This mode requests direct JSON output from Anthropic models. It:
- Works with all Claude models
- Is less structured than tool calling
- May be more flexible for simple cases

**Best for**: Simple structures or with older Claude models.

### Google/Gemini Modes

#### `GEMINI_TOOLS` Mode

```python
client = instructor.from_gemini(
    genai.GenerativeModel("gemini-1.5-pro"),
    mode=instructor.Mode.GEMINI_TOOLS
)
```

This mode uses Google's function calling for Gemini models. It:
- Provides structured tool definitions
- Works similarly to OpenAI's tool calling
- Has some Google-specific adaptations

**Best for**: Complex structured outputs with Gemini models.

**Note**: Cannot be used with multi-modal inputs.

#### `GEMINI_JSON` Mode

```python
client = instructor.from_gemini(
    genai.GenerativeModel("gemini-1.5-pro"),
    mode=instructor.Mode.GEMINI_JSON
)
```

This mode requests direct JSON from Gemini models. It:
- Works with all Gemini models
- Works with multi-modal inputs (images, audio)
- May be less reliable for complex structures

**Best for**: Simple structures or when using multi-modal inputs.

## Mode Selection Guidelines

### When to Use Tool-based Modes

- For complex, nested data structures
- When reliability is critical
- When working with modern models that support tool calling
- In production systems where validation is important

### When to Use JSON-based Modes

- For simple data structures
- When working with models that don't support tool calling
- When token efficiency is important
- When working with multi-modal inputs (for providers that don't support tools with multi-modal)

## Provider-Specific Recommendations

### OpenAI

```python
# Best for most OpenAI use cases
client = instructor.from_openai(OpenAI(), mode=instructor.Mode.TOOLS)

# For very simple extractions
client = instructor.from_openai(OpenAI(), mode=instructor.Mode.JSON)

# For complex, multi-step processes
client = instructor.from_openai(OpenAI(), mode=instructor.Mode.PARALLEL_TOOLS)
```

### Anthropic

```python
# For Claude 3+ with complex structures
client = instructor.from_anthropic(Anthropic(), mode=instructor.Mode.ANTHROPIC_TOOLS)

# For simpler extractions or older Claude models
client = instructor.from_anthropic(Anthropic(), mode=instructor.Mode.ANTHROPIC_JSON)
```

### Google/Gemini

```python
# For complex structures (non-multi-modal)
client = instructor.from_gemini(
    genai.GenerativeModel("gemini-1.5-pro"),
    mode=instructor.Mode.GEMINI_TOOLS
)

# For multi-modal inputs or simple structures
client = instructor.from_gemini(
    genai.GenerativeModel("gemini-1.5-pro"),
    mode=instructor.Mode.GEMINI_JSON
)
```

## Mode Compatibility Table

| Provider   | Tool-based Modes | JSON-based Modes |
|------------|------------------|------------------|
| OpenAI     | TOOLS, TOOLS_STRICT, PARALLEL_TOOLS, FUNCTIONS | JSON, MD_JSON, JSON_O1 |
| Anthropic  | ANTHROPIC_TOOLS | ANTHROPIC_JSON |
| Gemini     | GEMINI_TOOLS | GEMINI_JSON |
| Vertex AI  | VERTEXAI_TOOLS | VERTEXAI_JSON |
| Cohere     | COHERE_TOOLS | JSON, MD_JSON |
| Mistral    | MISTRAL_TOOLS | MISTRAL_STRUCTURED_OUTPUTS |
| Anyscale   | - | JSON, MD_JSON, JSON_SCHEMA |
| Databricks | TOOLS | JSON, MD_JSON |
| Together   | - | JSON, MD_JSON |
| Fireworks  | FIREWORKS_TOOLS | FIREWORKS_JSON |
| Cerebras   | - | CEREBRAS_JSON |
| Writer     | WRITER_TOOLS | JSON |
| Perplexity | - | PERPLEXITY_JSON |
| GenAI      | GENAI_TOOLS | GENAI_STRUCTURED_OUTPUTS |
| LiteLLM    | (depends on provider) | (depends on provider) |

## Best Practices

1. **Start with the recommended mode for your provider**
   - For OpenAI: `TOOLS`
   - For Anthropic: `ANTHROPIC_TOOLS` (Claude 3+) or `ANTHROPIC_JSON`
   - For Gemini: `GEMINI_TOOLS` or `GEMINI_JSON` for multi-modal

2. **Try JSON modes for simple structures or if you encounter issues**
   - JSON modes often work with simpler schemas
   - They may be more token-efficient
   - They work with more models

3. **Use provider-specific modes when available**
   - Provider-specific modes are optimized for that provider
   - They handle special cases and requirements

4. **Test and validate**
   - Different modes may perform differently for your specific use case
   - Always test with your actual data and models