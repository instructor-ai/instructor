---
date: 2025-05-07
authors:
    - jxnl 
categories:
    - tutorials
    - anthropic
    - structured-data
---

# Using Anthropic's Web Search with Instructor for Real-Time Data

Anthropic's new web search tool, when combined with Instructor, provides a powerful way to get real-time, structured data from the web. This allows you to build applications that can answer questions and provide information that is up-to-date, going beyond the knowledge cut-off of large language models.

In this post, we'll explore how to use the `web_search` tool with Instructor to fetch the latest information and structure it into a Pydantic model. Even a simple structure can be very effective for clarity and further processing.

<!-- more -->

## How it Works

The web search tool enables Claude models to perform web searches during a generation. When you provide the `web_search` tool in your API request, Claude can decide to use it if the prompt requires information it doesn't have. The API then executes the search, provides the results back to Claude, and Claude can then use this information to generate a response. Importantly, Claude will cite its sources from the search results. You can find more details in the [official Anthropic documentation](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/web-search-tool).

Instructor simplifies this process by allowing you to define a Pydantic model for the desired output structure. When Claude uses the web search tool and formulates an answer, Instructor ensures that the final output conforms to your defined schema.

## Example: Getting the Latest UFC Results

Let's look at a practical example. We want to get the latest UFC fight results.

First, ensure you have `instructor` and `anthropic` installed:

```bash
uv add instructor anthropic
```

Now, let's define our Pydantic model for the response:

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel

# Initialize the Anthropic client
# Make sure your ANTHROPIC_API_KEY environment variable is set
anthropic_client = Anthropic()

# Patch the client with Instructor
client = instructor.from_anthropic(anthropic_client, mode=instructor.Mode.ANTHROPIC_JSON)

class Respond(BaseModel):
    question: str
    answer: str
```

This `Respond` model is straightforward: it captures the original question and the textual answer provided by the LLM based on its web search.

Now, we can make the API call:

```python
# The rest of your imports and client setup from above...
# Make sure Respond is defined as above.

response_data, completion_details = client.messages.create_with_completion(
    model="claude-3-5-sonnet-latest", # Or other supported models like claude-3-7-sonnet
    max_tokens=1000, # Sufficient for a direct textual answer
    messages=[
        {"role": "user", "content": "What are the latest results for the UFC and who won?"}
    ],
    tools=[{
            "type": "web_search_20250305", # Correct tool type
            "name": "web_search",
            "max_uses": 3 # Limit the number of searches
        }
    ],
    response_model=Respond # Use the simple Respond model
)

print("Question:")
print(response_data.question)

print("\nAnswer:")
print(response_data.answer)
```

This approach provides a clean way to get the LLM's answer into a defined Pydantic object. The `examples/anthropic-web-tool/run.py` script reflects this implementation.

Expected output (will vary based on real-time web search data):

```
Question:
What are the latest results for the UFC and who won?

Answer:
The most recent UFC event was UFC Fight Night: Sandhagen vs Figueiredo on May 3, 2025, 
where Cory Sandhagen defeated Deiveson Figueiredo via TKO in the second round due to a knee injury. 
In the co-main event, Reinier de Ridder defeated Bo Nickal, and Daniel Rodriguez won against Santiago 
Ponzinibbio by third-round stoppage.
```

## Key Benefits

*   **Real-Time Information**: Access the latest data directly from the web.
*   **Structured Output**: Even with a simple model, Instructor ensures the output is a Pydantic object, making it easy to work with programmatically.
*   **Source Citations**: Claude automatically cites sources, allowing for verification (details in the API response, not shown in this simplified example).
*   **Reduced Hallucinations**: By relying on web search for factual, up-to-the-minute data, the likelihood of the LLM providing incorrect or outdated information is reduced.

## Configuring the Web Search Tool

Anthropic provides several options to configure the web search tool:

*   `max_uses`: Limit the number of searches Claude can perform in a single request.
*   `allowed_domains`: Restrict searches to a list of specific domains.
*   `blocked_domains`: Prevent searches on certain domains.
*   `user_location`: Localize search results by providing an approximate location (city, region, country, timezone).

For example, to limit searches to 3 and only allow results from `espn.com` and `ufc.com`:

```python
    tools=[{
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 3,
            "allowed_domains": ["espn.com", "ufc.com"]
        }
    ],
```

You cannot use `allowed_domains` and `blocked_domains` in the same request.

## Conclusion

Combining Anthropic's web search tool with Instructor's structured data capabilities opens up exciting possibilities for building dynamic, information-rich applications. Whether you're tracking sports scores, news updates, or market trends, this powerful duo can help you access and organize real-time web data effectively, even with simple Pydantic models.

Check out the example code in `examples/anthropic-web-tool/run.py` to see this implementation, and refer to the [Anthropic web search documentation](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/web-search-tool) for more in-depth information on the tool's capabilities.
