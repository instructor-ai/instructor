# Hooks

 Hooks let you intercept and handle events during completion and parsing in the Instructor library. Use hooks to add custom logging, metrics, or error handling at key stages of an API call.

## Basic Usage

```python
 import instructor
 from openai import OpenAI

 client = instructor.from_openai(OpenAI())

 def on_response(response):
     print("Raw response:", response)

 def on_error(error):
     print("Error occurred:", error)

 # Register hooks
 client.on("completion:response", on_response)
 client.on("completion:error", on_error)

 # Make a request
 result = client.chat.completions.create(
     model="gpt-3.5-turbo",
     messages=[{"role":"user","content":"Hello"}],
     response_model=str,
 )
 # 
```

 ## Hook Events

 - `completion:kwargs` — before sending to LLM
 - `completion:response` — after receiving LLM response
 - `completion:error` — on error before retry
 - `parse:error` — on parsing failure
 - `completion:last_attempt` — on final retry attempt

 ## Next Steps

 - Learn more in the core concept: [Hooks Concept](/concepts/hooks.md)
 - Explore advanced patterns: [Parallel Tool Calling](parallel_tool_calling.md)
 - Return to [Advanced Topics](index.md#advanced-topics)