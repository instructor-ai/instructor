# Phase 1: Resolve TODOs

**Status**: Not Started
**Priority**: P0 (Critical - unblocks features)
**Est. Duration**: 1-2 weeks
**Est. Effort**: 3-5 days
**Assignee**: TBD
**Dependencies**: None

---

## Quick Reference

| TODO | File | Line | Effort | Priority |
|------|------|------|--------|----------|
| Batch API (×8) | anthropic.py, batch.py | Various | 1 day | P0 |
| Anthropic types | function_calls.py | 340 | 2-4h | P1 |
| Cohere content | function_calls.py | 302 | 6-12h | P1 |
| VertexAI investigation | vertexai/client.py | 25 | 8-20h | P2 |
| Type system | distil.py | 187 | 4-8h | P3 |
| Test skip | test_validators.py | 28 | Defer | P4 |

---

## Overview

Resolve all 12 TODOs currently in the codebase. Eight TODOs follow an identical pattern and can be fixed with a single helper function. The remaining TODOs require individual attention.

### Goals

1. Eliminate all **8 Anthropic Batch API TODOs** with a single refactor
2. Fix **2 content type handling TODOs** in function_calls.py
3. Investigate and document **VertexAI iterable fields issue**
4. Fix **type system TODO** in distil.py
5. Document or defer **test skip TODO**

### Success Metrics

- ☐ 12 → 0 TODOs (or documented deferrals)
- ☐ ~40 lines of code duplication eliminated
- ☐ All batch API operations use consistent pattern
- ☐ Content type handling supports multimodal inputs
- ☐ VertexAI issue documented with reproduction case

---

## Current State Analysis

### TODO Distribution

**By File**:
- `instructor/batch/providers/anthropic.py`: 7 TODOs
- `instructor/cli/batch.py`: 1 TODO
- `instructor/processing/function_calls.py`: 2 TODOs
- `instructor/providers/vertexai/client.py`: 1 TODO
- `instructor/distil.py`: 1 TODO
- `tests/llm/test_openai/test_validators.py`: 1 TODO

**By Category**:
- Batch API workarounds: 8 TODOs (67%)
- Content type handling: 2 TODOs (17%)
- Investigation needed: 1 TODO (8%)
- Type system: 1 TODO (8%)

---

## Task 1: Anthropic Batch API (8 TODOs)

**Priority**: P0 (highest)
**Effort**: 1 day
**Impact**: Eliminates most TODOs, ~40 lines of duplication

### Problem

All 8 TODOs use identical beta API fallback pattern:

**Locations**:
1. `instructor/batch/providers/anthropic.py:40`
2. `instructor/batch/providers/anthropic.py:76`
3. `instructor/batch/providers/anthropic.py:99`
4. `instructor/batch/providers/anthropic.py:146`
5. `instructor/batch/providers/anthropic.py:191`
6. `instructor/batch/providers/anthropic.py:209`
7. `instructor/batch/providers/anthropic.py:231`
8. `instructor/cli/batch.py:395`

**Current Code** (repeated 8 times):
```python
# TODO(#batch-api-stable): Remove this once batch API is stable
try:
    result = client.messages.batches.create(...)
except AttributeError:
    # Fallback to beta API
    result = client.beta.messages.batches.create(...)
```

### Solution

Create a single helper function to abstract the beta fallback:

**File**: `instructor/batch/providers/anthropic.py`

```python
def _batch_api_call(client, method_name: str, **kwargs):
    """
    Handle Anthropic batch API calls with automatic beta fallback.

    The Anthropic batch API is transitioning from beta to stable. This helper
    attempts the stable API first, falling back to beta if unavailable.

    Args:
        client: Anthropic client instance
        method_name: Name of the batch method ('create', 'retrieve', 'cancel', etc.)
        **kwargs: Arguments to pass to the batch method

    Returns:
        Result from the batch API call

    Raises:
        AttributeError: If neither stable nor beta API is available
    """
    try:
        # Try stable API first
        method = getattr(client.messages.batches, method_name)
        return method(**kwargs)
    except AttributeError:
        # Fallback to beta API
        try:
            method = getattr(client.beta.messages.batches, method_name)
            return method(**kwargs)
        except AttributeError:
            raise AttributeError(
                f"Anthropic client does not support batch API method '{method_name}'. "
                f"Ensure you have the latest anthropic package installed: "
                f"pip install --upgrade anthropic"
            )
```

### Implementation Steps

#### Step 1: Add Helper Function (15 minutes)

1. Open `instructor/batch/providers/anthropic.py`
2. Add the `_batch_api_call()` helper function at the top of the file (after imports, before other functions)
3. Add comprehensive docstring explaining the beta fallback

#### Step 2: Replace All Usages (30 minutes)

**Location 1**: Line 40 (create_batch method)
```python
# Before
try:
    batch = client.messages.batches.create(...)
except AttributeError:
    batch = client.beta.messages.batches.create(...)

# After
batch = _batch_api_call(client, "create", ...)
```

**Location 2**: Line 76 (retrieve_batch method)
```python
# Before
try:
    batch = client.messages.batches.retrieve(batch_id)
except AttributeError:
    batch = client.beta.messages.batches.retrieve(batch_id)

# After
batch = _batch_api_call(client, "retrieve", id=batch_id)
```

**Location 3**: Line 99 (cancel_batch method)
```python
# Before
try:
    result = client.messages.batches.cancel(batch_id)
except AttributeError:
    result = client.beta.messages.batches.cancel(batch_id)

# After
result = _batch_api_call(client, "cancel", id=batch_id)
```

**Location 4**: Line 146 (list_batches method)
```python
# Before
try:
    batches = client.messages.batches.list(...)
except AttributeError:
    batches = client.beta.messages.batches.list(...)

# After
batches = _batch_api_call(client, "list", ...)
```

**Locations 5-7**: Lines 191, 209, 231 (similar pattern)
Apply the same transformation using `_batch_api_call`.

**Location 8**: `instructor/cli/batch.py:395`
```python
# Before
try:
    result = client.messages.batches.retrieve(batch_id)
except AttributeError:
    result = client.beta.messages.batches.retrieve(batch_id)

# After
from instructor.batch.providers.anthropic import _batch_api_call
result = _batch_api_call(client, "retrieve", id=batch_id)
```

#### Step 3: Test (1 hour)

**Test both code paths**:

1. **Test stable API** (if available):
```bash
uv run pytest tests/llm/test_anthropic/test_batch.py -v
```

2. **Test beta fallback**:
Mock the stable API to raise AttributeError:
```python
# tests/llm/test_anthropic/test_batch_fallback.py
import pytest
from unittest.mock import Mock, patch
from instructor.batch.providers.anthropic import _batch_api_call

def test_batch_api_stable():
    """Test that stable API is used when available."""
    client = Mock()
    client.messages.batches.create.return_value = {"id": "batch_123"}

    result = _batch_api_call(client, "create", param1="value1")

    assert result == {"id": "batch_123"}
    client.messages.batches.create.assert_called_once_with(param1="value1")
    client.beta.messages.batches.create.assert_not_called()

def test_batch_api_beta_fallback():
    """Test that beta API is used when stable API unavailable."""
    client = Mock()
    client.messages.batches = Mock(spec=[])  # No 'create' attribute
    client.beta.messages.batches.create.return_value = {"id": "batch_456"}

    result = _batch_api_call(client, "create", param1="value1")

    assert result == {"id": "batch_456"}
    client.beta.messages.batches.create.assert_called_once_with(param1="value1")

def test_batch_api_no_support():
    """Test error message when neither API is available."""
    client = Mock()
    client.messages.batches = Mock(spec=[])
    client.beta.messages.batches = Mock(spec=[])

    with pytest.raises(AttributeError, match="does not support batch API"):
        _batch_api_call(client, "create", param1="value1")
```

Run tests:
```bash
uv run pytest tests/llm/test_anthropic/test_batch_fallback.py -v
```

#### Step 4: Update Documentation (15 minutes)

Add comment explaining the transition:

```python
# instructor/batch/providers/anthropic.py (at top of file)
"""
Anthropic Batch API Provider

The Anthropic batch API is transitioning from beta to stable. All batch operations
use the _batch_api_call() helper which automatically handles fallback to the beta
API if the stable API is not yet available in the installed anthropic package.

Once the batch API is fully stable and widely adopted, the beta fallback can be
removed from _batch_api_call().
"""
```

### Rollback Plan

If issues are discovered:

1. **Revert the commits**:
```bash
git revert <commit-hash>
```

2. **TODOs will return**: The original try/except pattern is functional, just verbose

3. **No data loss**: This change only affects API access pattern, not data handling

### Success Criteria

- ☐ Helper function `_batch_api_call()` created
- ☐ All 8 TODOs replaced with helper calls
- ☐ Tests pass for both stable and beta APIs
- ☐ Documentation updated
- ☐ ~40 lines of duplication eliminated

---

## Task 2a: Anthropic Type Checking (1 TODO)

**Priority**: P1
**Effort**: 2-4 hours
**Impact**: More robust content handling

### Problem

**Location**: `instructor/processing/function_calls.py:340`

**Current Code**:
```python
# TODO: replace string comparison with proper type checking
if content.get("type") == "image":
    # Handle image content
    pass
```

**Issue**: Using string comparison instead of type guards. Fragile and error-prone.

### Solution

Use proper type checking with TypedDict or Protocol:

```python
from typing import TypedDict, Literal

class ImageContent(TypedDict):
    type: Literal["image"]
    source: dict[str, Any]

class TextContent(TypedDict):
    type: Literal["text"]
    text: str

ContentPart = ImageContent | TextContent

def is_image_content(content: dict) -> TypeGuard[ImageContent]:
    """Type guard for image content."""
    return content.get("type") == "image" and "source" in content

def is_text_content(content: dict) -> TypeGuard[TextContent]:
    """Type guard for text content."""
    return content.get("type") == "text" and "text" in content
```

**Then use**:
```python
if is_image_content(content):
    # Type checker knows content is ImageContent here
    handle_image(content["source"])
elif is_text_content(content):
    # Type checker knows content is TextContent here
    handle_text(content["text"])
```

### Implementation Steps

#### Step 1: Define Types (30 minutes)

Create `instructor/processing/content_types.py`:

```python
from typing import TypedDict, Literal, TypeGuard, Any

class ImageContent(TypedDict):
    """Anthropic image content block."""
    type: Literal["image"]
    source: dict[str, Any]  # More specific: ImageSource TypedDict

class TextContent(TypedDict):
    """Anthropic text content block."""
    type: Literal["text"]
    text: str

class AudioContent(TypedDict):
    """Anthropic audio content block (future)."""
    type: Literal["audio"]
    source: dict[str, Any]

ContentBlock = ImageContent | TextContent | AudioContent

def is_image_content(content: dict) -> TypeGuard[ImageContent]:
    """Type guard to check if content is an image block."""
    return content.get("type") == "image" and "source" in content

def is_text_content(content: dict) -> TypeGuard[TextContent]:
    """Type guard to check if content is a text block."""
    return content.get("type") == "text" and "text" in content

def is_audio_content(content: dict) -> TypeGuard[AudioContent]:
    """Type guard to check if content is an audio block."""
    return content.get("type") == "audio" and "source" in content
```

#### Step 2: Update function_calls.py (1 hour)

**Location**: Line 340

```python
# Before
def handle_anthropic_content(content: dict) -> dict:
    # TODO: replace string comparison with proper type checking
    if content.get("type") == "image":
        # Handle image
        pass
    elif content.get("type") == "text":
        # Handle text
        pass
    return processed_content

# After
from instructor.processing.content_types import (
    is_image_content,
    is_text_content,
    is_audio_content,
)

def handle_anthropic_content(content: dict) -> dict:
    if is_image_content(content):
        # Type checker knows content is ImageContent
        return handle_image_content(content["source"])
    elif is_text_content(content):
        # Type checker knows content is TextContent
        return handle_text_content(content["text"])
    elif is_audio_content(content):
        return handle_audio_content(content["source"])
    else:
        raise ValueError(
            f"Unsupported content type: {content.get('type')}. "
            f"Supported types: text, image, audio"
        )
```

#### Step 3: Add Tests (1 hour)

```python
# tests/processing/test_content_types.py
import pytest
from instructor.processing.content_types import (
    is_image_content,
    is_text_content,
    is_audio_content,
)

def test_is_image_content():
    assert is_image_content({"type": "image", "source": {...}})
    assert not is_image_content({"type": "text", "text": "hello"})
    assert not is_image_content({"type": "image"})  # Missing source

def test_is_text_content():
    assert is_text_content({"type": "text", "text": "hello"})
    assert not is_text_content({"type": "image", "source": {...}})
    assert not is_text_content({"type": "text"})  # Missing text

def test_is_audio_content():
    assert is_audio_content({"type": "audio", "source": {...}})
    assert not is_audio_content({"type": "text", "text": "hello"})
```

### Success Criteria

- ☐ `content_types.py` created with type guards
- ☐ String comparisons replaced with type guards
- ☐ Type checker passes (`uv run ty check`)
- ☐ Tests added and passing

---

## Task 2b: Cohere Content Types (1 TODO)

**Priority**: P1
**Effort**: 6-12 hours
**Impact**: Enables multimodal support for Cohere V2

### Problem

**Location**: `instructor/processing/function_calls.py:302`

**Current Code**:
```python
# TODO: Handle other content types (image, audio, etc)
if isinstance(part, str):
    content.append({"type": "text", "text": part})
# Missing: image, audio, file handling
```

**Issue**: Only handles text content. Need image/audio/file support for Cohere V2.

### Solution

Add full multimodal content handling:

```python
def handle_cohere_content_part(part: Any) -> dict:
    """Convert a content part to Cohere V2 format."""
    if isinstance(part, str):
        return {"type": "text", "text": part}
    elif isinstance(part, dict):
        content_type = part.get("type")
        if content_type == "image":
            return handle_cohere_image(part)
        elif content_type == "audio":
            return handle_cohere_audio(part)
        elif content_type == "file":
            return handle_cohere_file(part)
        else:
            raise ValueError(f"Unsupported content type: {content_type}")
    else:
        raise TypeError(f"Unsupported content part type: {type(part)}")
```

### Implementation Steps

#### Step 1: Research Cohere V2 API (1-2 hours)

1. Read Cohere V2 API docs for multimodal support
2. Identify supported content types
3. Document expected input/output formats

**Documentation locations**:
- https://docs.cohere.com/v2/docs/multimodal-models
- Check `cohere` package source code for type definitions

#### Step 2: Implement Content Handlers (3-4 hours)

```python
# instructor/processing/function_calls.py

def handle_cohere_image(image_part: dict) -> dict:
    """
    Convert image content to Cohere V2 format.

    Accepts:
    - {"type": "image", "source": "url", "url": "https://..."}
    - {"type": "image", "source": "base64", "data": "..."}
    """
    if image_part.get("source") == "url":
        return {
            "type": "image",
            "source": {
                "type": "url",
                "url": image_part["url"]
            }
        }
    elif image_part.get("source") == "base64":
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "data": image_part["data"],
                "media_type": image_part.get("media_type", "image/jpeg")
            }
        }
    else:
        raise ValueError(f"Unsupported image source: {image_part.get('source')}")

def handle_cohere_audio(audio_part: dict) -> dict:
    """Convert audio content to Cohere V2 format."""
    # Similar structure to handle_cohere_image
    # Implementation depends on Cohere V2 audio API
    raise NotImplementedError("Cohere V2 audio support not yet implemented")

def handle_cohere_file(file_part: dict) -> dict:
    """Convert file content to Cohere V2 format."""
    # Implementation depends on Cohere V2 file API
    raise NotImplementedError("Cohere V2 file support not yet implemented")
```

#### Step 3: Update Main Handler (1 hour)

Replace TODO at line 302:

```python
# Before
def cohere_process_content(content: list) -> list:
    result = []
    for part in content:
        # TODO: Handle other content types
        if isinstance(part, str):
            result.append({"type": "text", "text": part})
    return result

# After
def cohere_process_content(content: list) -> list:
    result = []
    for part in content:
        result.append(handle_cohere_content_part(part))
    return result

def handle_cohere_content_part(part: Any) -> dict:
    """Convert content part to Cohere V2 format with full multimodal support."""
    if isinstance(part, str):
        return {"type": "text", "text": part}
    elif isinstance(part, dict):
        content_type = part.get("type")
        if content_type == "text":
            return {"type": "text", "text": part["text"]}
        elif content_type == "image":
            return handle_cohere_image(part)
        elif content_type == "audio":
            return handle_cohere_audio(part)
        elif content_type == "file":
            return handle_cohere_file(part)
        else:
            raise ValueError(
                f"Unsupported Cohere content type: {content_type}. "
                f"Supported types: text, image"
            )
    else:
        raise TypeError(
            f"Cohere content part must be str or dict, got {type(part)}"
        )
```

#### Step 4: Add Tests (2-3 hours)

```python
# tests/processing/test_cohere_content.py
import pytest
from instructor.processing.function_calls import (
    handle_cohere_content_part,
    handle_cohere_image,
)

def test_cohere_text_string():
    result = handle_cohere_content_part("Hello world")
    assert result == {"type": "text", "text": "Hello world"}

def test_cohere_text_dict():
    result = handle_cohere_content_part({"type": "text", "text": "Hello"})
    assert result == {"type": "text", "text": "Hello"}

def test_cohere_image_url():
    part = {
        "type": "image",
        "source": "url",
        "url": "https://example.com/image.jpg"
    }
    result = handle_cohere_content_part(part)
    assert result["type"] == "image"
    assert result["source"]["type"] == "url"
    assert result["source"]["url"] == "https://example.com/image.jpg"

def test_cohere_image_base64():
    part = {
        "type": "image",
        "source": "base64",
        "data": "iVBORw0KG...",
        "media_type": "image/png"
    }
    result = handle_cohere_content_part(part)
    assert result["type"] == "image"
    assert result["source"]["type"] == "base64"
    assert result["source"]["data"] == "iVBORw0KG..."

def test_cohere_unsupported_type():
    with pytest.raises(ValueError, match="Unsupported Cohere content type"):
        handle_cohere_content_part({"type": "video"})

def test_cohere_invalid_part_type():
    with pytest.raises(TypeError, match="must be str or dict"):
        handle_cohere_content_part(12345)
```

Run integration tests with actual Cohere API:
```bash
uv run pytest tests/llm/test_cohere/test_multimodal.py -v
```

### Success Criteria

- ☐ Image content handling implemented
- ☐ Audio/file handlers stubbed (NotImplementedError with clear message)
- ☐ All unit tests passing
- ☐ Integration tests with Cohere API passing
- ☐ Documentation added explaining supported formats

---

## Task 3: VertexAI Investigation (1 TODO)

**Priority**: P2
**Effort**: 8-20 hours (investigation)
**Impact**: Unblocks VertexAI iterable support or documents limitation

### Problem

**Location**: `instructor/providers/vertexai/client.py:25`

**Current Code**:
```python
# TODO: Figure out why vertexai needs required fields for iterable
def handle_vertexai_tools(...):
    if is_iterable(response_model):
        # Workaround: make all fields optional
        response_model = make_optional_all_fields(response_model)
```

**Issue**: VertexAI rejects schemas with optional iterable fields. Root cause unknown.

### Investigation Plan

#### Step 1: Reproduce the Issue (2-3 hours)

Create minimal reproduction case:

```python
# tests/llm/test_vertexai/test_iterable_investigation.py
import pytest
from pydantic import BaseModel
from instructor import from_vertexai
import google.generativeai as genai

class Item(BaseModel):
    name: str
    value: int

class ItemList(BaseModel):
    items: list[Item]

def test_vertexai_iterable_with_optional_fields():
    """Test if VertexAI accepts schemas with optional iterable fields."""
    client = from_vertexai(genai.GenerativeModel("gemini-1.5-pro"))

    # This should fail with current implementation
    result = client.chat.completions.create(
        messages=[{"role": "user", "content": "List 3 items"}],
        response_model=Iterable[Item],
    )

    items = list(result)
    assert len(items) == 3

def test_vertexai_iterable_schema_validation():
    """Examine the schema sent to VertexAI."""
    from instructor.providers.vertexai.utils import handle_vertexai_tools
    from instructor.dsl.iterable import Iterable

    # Generate schema
    response_model, kwargs = handle_vertexai_tools(
        response_model=Iterable[Item],
        kwargs={}
    )

    # Inspect the schema
    print("Generated schema:", kwargs.get("tools"))

    # Check if fields are optional
    # Document findings
```

Run and capture output:
```bash
uv run pytest tests/llm/test_vertexai/test_iterable_investigation.py -v -s
```

#### Step 2: Compare with Working Providers (2-3 hours)

Test the same schema with other providers:

```python
# Compare schema generation across providers
providers_to_test = [
    ("OpenAI", from_openai),
    ("Anthropic", from_anthropic),
    ("VertexAI", from_vertexai),
]

for provider_name, factory_func in providers_to_test:
    print(f"\n{provider_name} Schema:")
    # Generate schema
    # Print and compare
```

**Questions to answer**:
- Do other providers handle Iterable differently?
- What schema does OpenAI/Anthropic generate for Iterable?
- What specific error does VertexAI return?

#### Step 3: Examine VertexAI API Docs (2-4 hours)

1. Search VertexAI documentation for schema requirements
2. Check if there are known limitations with optional fields
3. Look for VertexAI-specific schema constraints
4. Review VertexAI Python client source code

**Resources**:
- https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/function-calling
- VertexAI Python client GitHub: https://github.com/googleapis/python-aiplatform

#### Step 4: Test Hypotheses (3-5 hours)

**Hypothesis 1**: VertexAI requires all tool parameters to be required
```python
# Test with all required fields
class StrictItem(BaseModel):
    name: str  # required
    value: int  # required
```

**Hypothesis 2**: VertexAI doesn't support array types in tool schemas
```python
# Test with single object instead of array
response_model=Item  # Not Iterable[Item]
```

**Hypothesis 3**: VertexAI has different JSON schema dialect
```python
# Test with explicit schema configuration
# Examine schema format differences
```

#### Step 5: Document Findings (2-3 hours)

**If root cause found**:
Create `docs/providers/vertexai_limitations.md`:

```markdown
# VertexAI Limitations

## Iterable Fields Require Workaround

**Issue**: VertexAI's function calling API rejects schemas where iterable fields
have optional properties.

**Root Cause**: [Document the actual cause found]

**Workaround**: The `handle_vertexai_tools()` function automatically makes all
fields optional when using `Iterable[T]` response models.

**Impact**: Validation is less strict for VertexAI iterables.

**Example**:
[Provide working example]

**Upstream Issue**: [Link to VertexAI bug if filed]
```

**If root cause NOT found**:
Document the investigation and workaround:

```markdown
# VertexAI Iterable Fields Investigation

**Status**: Root cause unknown (investigated [date])

**Symptom**: Schemas with optional iterable fields cause errors

**Workaround**: Use `make_optional_all_fields()` (current implementation)

**Investigation Results**:
- [Summarize findings]
- [What was tested]
- [What doesn't work]

**Next Steps**:
- File bug with Google VertexAI team
- Continue monitoring for API updates
- Consider alternative approaches:
  1. [List alternatives]
```

### Success Criteria

- ☐ Minimal reproduction case created
- ☐ Issue compared across providers
- ☐ VertexAI docs thoroughly reviewed
- ☐ All reasonable hypotheses tested
- ☐ Findings documented (either root cause or investigation results)
- ☐ Bug filed with VertexAI if appropriate
- ☐ Workaround validated or improved

---

## Task 4: Type System Fix (1 TODO)

**Priority**: P3
**Effort**: 4-8 hours
**Impact**: Better type safety in distil.py

### Problem

**Location**: `instructor/distil.py:187`

**Current Code**:
```python
# TODO: Fix type checker - doesn't recognize response_model parameter
response = client.chat.completions.create(  # type: ignore
    response_model=response_model,
    ...
)
```

**Issue**: Type checker doesn't recognize `response_model` as a valid parameter.

### Solution

Add proper type stubs or overload signatures:

**Option A**: Type stub file
```python
# instructor/client.pyi
from typing import overload, TypeVar
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)

class ChatCompletions:
    @overload
    def create(
        self,
        *,
        model: str,
        messages: list[dict],
        response_model: type[T],
        **kwargs
    ) -> T: ...

    @overload
    def create(
        self,
        *,
        model: str,
        messages: list[dict],
        **kwargs
    ) -> Any: ...
```

**Option B**: Update function signature
```python
# instructor/client.py
def create(
    self,
    *,
    model: str,
    messages: list[dict],
    response_model: type[BaseModel] | None = None,
    **kwargs
) -> BaseModel | Any:
    ...
```

### Implementation Steps

#### Step 1: Identify Root Cause (1-2 hours)

1. Reproduce the type checker error:
```bash
uv run ty check instructor/distil.py
```

2. Examine the type checker output
3. Check if `response_model` is defined in the signature
4. Review how other files handle this parameter

#### Step 2: Choose Solution Approach (30 minutes)

**If response_model is missing from signature**:
- Option A: Add it to the function signature

**If signature is correct but type checker confused**:
- Option B: Add type stub file

**If due to dynamic patching**:
- Option C: Add Protocol or type cast

#### Step 3: Implement Fix (2-3 hours)

**For Option A** (update signature):
```python
# instructor/client.py
from typing import TypeVar, overload
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)

class ChatCompletions:
    @overload
    def create(
        self,
        *,
        response_model: type[T],
        **kwargs
    ) -> T: ...

    @overload
    def create(
        self,
        *,
        response_model: None = None,
        **kwargs
    ) -> Any: ...

    def create(self, **kwargs):
        # Implementation
        ...
```

**For Option B** (type stub):
Create `instructor/client.pyi` with proper overloads.

**For Option C** (Protocol):
```python
# instructor/protocols.py
from typing import Protocol, TypeVar
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)

class CompletionClient(Protocol):
    def create(
        self,
        *,
        response_model: type[T],
        **kwargs
    ) -> T: ...
```

#### Step 4: Remove type: ignore (15 minutes)

In `instructor/distil.py`:
```python
# Before
response = client.chat.completions.create(  # type: ignore
    response_model=response_model,
    ...
)

# After
response = client.chat.completions.create(
    response_model=response_model,
    ...
)
```

#### Step 5: Verify (30 minutes)

```bash
# Type check should pass now
uv run ty check instructor/distil.py

# Run tests to ensure no runtime issues
uv run pytest tests/test_distil.py -v
```

### Success Criteria

- ☐ Type checker error resolved
- ☐ `# type: ignore` removed
- ☐ Type stubs or overloads added
- ☐ All tests passing
- ☐ No regressions in other files

---

## Task 5: Test Skip Documentation (1 TODO)

**Priority**: P4 (defer)
**Effort**: Unknown
**Impact**: Low (test-only)

### Problem

**Location**: `tests/llm/test_openai/test_validators.py:28`

**Current Code**:
```python
# TODO: Figure out why this test needs to be skipped sometimes
@pytest.mark.skip(reason="Flaky test - needs investigation")
def test_validator_with_retry():
    ...
```

### Recommended Action

**Defer this TODO**:
- It's test-only, doesn't affect production code
- Requires investigation to understand flakiness
- Low priority compared to other TODOs

**Document for future**:
```python
# TODO(deferred): Investigate test flakiness
# This test occasionally fails in CI. Possible causes:
# - Rate limiting
# - Non-deterministic model outputs
# - Race condition in retry logic
# See: [link to issue when created]
@pytest.mark.skip(reason="Flaky - deferred to Phase 1.6 (Test Improvements)")
def test_validator_with_retry():
    ...
```

**Plan for Phase 6** (Test Improvements):
- Investigate flakiness systematically
- Add retry logic to flaky tests
- Or mark as `@pytest.mark.flaky(reruns=3)` if using pytest-rerunfailures

### Success Criteria

- ☐ TODO documented as deferred
- ☐ Issue created for tracking
- ☐ Planned for Phase 6 investigation

---

## Testing Strategy

### Unit Tests

```bash
# Test batch API helper
uv run pytest tests/llm/test_anthropic/test_batch_fallback.py -v

# Test content type handlers
uv run pytest tests/processing/test_content_types.py -v
uv run pytest tests/processing/test_cohere_content.py -v

# Test type system fix
uv run ty check instructor/distil.py
```

### Integration Tests

```bash
# Test Anthropic batch operations
uv run pytest tests/llm/test_anthropic/test_batch.py -v

# Test Cohere multimodal
uv run pytest tests/llm/test_cohere/test_multimodal.py -v

# Test VertexAI iterables
uv run pytest tests/llm/test_vertexai/test_iterable_investigation.py -v
```

### Regression Tests

```bash
# Full test suite
uv run pytest tests/ -v

# Provider-specific tests
uv run pytest tests/llm/test_anthropic/ -v
uv run pytest tests/llm/test_cohere/ -v
uv run pytest tests/llm/test_vertexai/ -v
```

---

## Timeline

### Day 1: Anthropic Batch API
- Morning: Implement `_batch_api_call()` helper
- Afternoon: Replace all 8 usages, test both code paths

### Day 2-3: Content Type Handling
- Day 2 Morning: Anthropic type checking (Task 2a)
- Day 2 Afternoon: Research Cohere V2 multimodal API
- Day 3: Implement Cohere content handlers (Task 2b)

### Day 4-5: VertexAI Investigation
- Day 4: Reproduce issue, compare providers, review docs
- Day 5: Test hypotheses, document findings

### Day 6: Type System & Wrap-up
- Morning: Fix type system TODO (Task 4)
- Afternoon: Document test skip (Task 5), final testing

### Day 7: Buffer
- Address any issues found during testing
- Update documentation
- Prepare summary of changes

---

## Rollback Plan

### If Batch API Changes Cause Issues

```bash
# Revert specific commits
git revert <batch-api-commit-hash>

# TODOs will return but functionality is preserved
```

### If Content Type Changes Break Providers

```bash
# Revert content type changes
git revert <content-types-commit-hash>

# Old string comparison will work (less type-safe but functional)
```

### If VertexAI Investigation Breaks Iterable

```bash
# Keep existing workaround, just update documentation
# No code changes needed if investigation doesn't find solution
```

---

## Communication Plan

### Daily Updates

Post to team channel:
- What was completed today
- Any blockers encountered
- Plan for tomorrow

### Weekly Summary

After week 1:
- Summary of TODOs resolved (should be 8-10 of 12)
- Metrics: lines eliminated, tests added
- Remaining work for week 2

---

## Success Criteria for Phase 1

Phase 1 is **complete** when:

- ☐ 11 of 12 TODOs resolved (1 deferred to Phase 6)
- ☐ All batch API operations use `_batch_api_call()` helper
- ☐ Content type handling supports text + image (audio/file stubbed)
- ☐ VertexAI issue is documented with reproduction case
- ☐ Type system TODO resolved or documented
- ☐ Test skip TODO documented as deferred
- ☐ All tests passing
- ☐ ~40 lines of duplication eliminated
- ☐ Documentation updated
- ☐ No regressions introduced

---

## Next Phase

Once Phase 1 is complete, proceed to:
- **[Phase 2: Provider Base Classes](./phase2_base_classes.md)** - can start immediately in parallel
- **[Phase 3: Consolidate Retry Logic](./phase3_retry_consolidation.md)** - independent, can start anytime
- **[Phase 4: Type System Improvements](./phase4_type_system.md)** - builds on Phase 1

---

**Status**: Ready to start
**Assignee**: TBD
**Start Date**: TBD
**Target Completion**: TBD + 1-2 weeks
