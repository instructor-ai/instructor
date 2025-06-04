# Verbose Control

The instructor library provides a `verbose` parameter to control output verbosity, particularly useful when working with schema generation that might produce unwanted console output.

## Gemini Verbose Control

When using the Gemini provider with `GEMINI_TOOLS` mode, you can control whether schema information is printed to the console:

```python
import instructor
import google.generativeai as genai

client = genai.GenerativeModel("gemini-pro")

# Suppress schema output (verbose=False)
instructor_client = instructor.from_gemini(
    client=client,
    mode=instructor.Mode.GEMINI_TOOLS,
    verbose=False,  # Prevents schema from being printed
)

# Default behavior (verbose=True)
instructor_client = instructor.from_gemini(
    client=client,
    mode=instructor.Mode.GEMINI_TOOLS,
    verbose=True,   # Allows schema to be printed (default)
)
```

## Use Cases

The verbose control is particularly useful in:

- Production environments where you want clean logs
- Automated scripts where schema output is not needed
- Applications with sensitive schema information
- Batch processing where excessive output slows down execution

## Default Behavior

By default, `verbose=True` to maintain backward compatibility. This means existing code will continue to work as before unless you explicitly set `verbose=False`.

## Implementation Details

When `verbose=False`, the library uses context managers to suppress stdout and stderr output during schema generation, ensuring that the functionality remains intact while preventing unwanted console output.
