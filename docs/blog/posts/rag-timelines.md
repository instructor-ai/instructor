---
draft: False
date: 2024-02-17
tags:
  - RAG
authors:
  - jxnl
---

# Enhancing RAG with Time Filters Using Instructor

Retrieval-augmented generation (RAG) systems often need to handle queries with time-based constraints, like "What new features were released last quarter?" or "Show me support tickets from the past week." Effective time filtering is crucial for providing accurate, relevant responses.

Instructor is a Python library that simplifies integrating large language models (LLMs) with data sources and APIs. It allows defining structured output models using Pydantic, which can be used as prompts or to parse LLM outputs.

## Modeling Time Filters

To handle time filters, we can define a Pydantic model representing a time range:

```python
from datetime import datetime
from typing import Optional
from pydantic import BaseModel

class TimeFilter(BaseModel):
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
```

The `TimeFilter` model can represent an absolute date range or a relative time range like "last week" or "previous month."

We can then combine this with a search query string:

```python
class SearchQuery(BaseModel):
    query: str
    time_filter: TimeFilter
```

## Prompting the LLM

Using Instructor, we can prompt the LLM to generate a `SearchQuery` object based on the user's query:

```python
import instructor
from openai import OpenAI

client = instructor.from_openai(OpenAI())

response = client.chat.completions.create(
    model="gpt-4o",
    response_model=SearchQuery,
    messages=[
        {
            "role": "system", 
            "content": "You are a query generator for customer support tickets. The current date is 2024-02-17"},
        {
            "role": "user", 
            "content": "Show me customer support tickets opened in the past week."
        },
    ],
)

{
    "query": "Show me customer support tickets opened in the past week.",
    "time_filter": {
        "start_date": "2024-02-10T00:00:00",
        "end_date": "2024-02-17T00:00:00"
    }
}
```

The LLM will generate a search query string and a `TimeFilter` object representing "past week" which you can use to filter the data in a subsequent request.

By modeling time filters with Pydantic and leveraging Instructor, RAG systems can effectively handle time-based queries. Clear prompts, careful model design, and appropriate parsing strategies enable accurate retrieval of information within specific time frames, enhancing the system's overall relevance and accuracy.