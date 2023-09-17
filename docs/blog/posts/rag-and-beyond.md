---
draft: True 
date: 2023-09-17
tags:
  - RAG
  - Embeddings
  - Query Understanding
  - Search Systems
  - Personal Assistant
---

# RAG: It's More Than Just Embeddings

Embeddings are just the tip of the iceberg when it comes to retrival augmented generation (RAG) with LLMS where you attach a 'knowledge base' to a language model. In this post I'll show you how to use `instructor` to model a search backend.

Throughout my consulting I've found that query understanding and building a proper suite of search tools has been critial. Once you have that in place, then the role of the LLM is to translate the user query and attaching that to a powerful search backend, or summarizing the results of that search backend back ot the user. However in practice, I'm 

Today I'll show you how to use `instructor` to model a search backend.

## The 'Dumb' RAG Model

When you ask a question like, "what is the capital of France?" The RAG 'dumb' model embeds the query and searches in some unopinonated search endpoint. Limited to a single method API like `search(query: str) -> List[str]`. This is fine for simple queries, since you'd expect words like 'paris is the capital of france' to be in the top results of say, your wikipedia embeddings.

However it'd be crazy to think that this was the way, its common to want to filter keywords, or tags, or search in a specific date range, or dispatch to multiple backends. This is where modeling your search backend comes in.

## Case Study 1: Metaphor Systems

Take Metaphor Systems, which turns natural language queries into search-optimized queries. I you see their web ui you'll notice that they have an auto-prompt option, which uses function calls to take your query and turn it into a fuller specified search query that is aware of the search backend.

```python
# MetaphorQuery model and OpenAI call
from pydantic import BaseModel
import datetime
from typing import List

class DateRange(BaseModel):
    start: datetime.date
    end: datetime.date

class MetaphorQuery(BaseModel):
    query: str
    published_daterange: DateRange
    domains_allow_list: List[str]

    async def execute():
        return await metaphor.search(
            query=self.query,
            published_daterange=self.published_daterange,
            domains_allow_list=self.domains_allow_list
        )

query = openai.ChatCompletion.create(
    model="gpt-4",
    response_model=MetaphorQuery,
    messages=[
        {
            "role": "system", 
            "content": "You're a query understanding system for the Metafor Systems search engine. Here are some tips: ..."
        },
        {
            "role": "user", 
            "content": "What are some recent developments in AI?"
        }
    ],
)
```

**Example Output**

```json
{
    "query": "What are some recent developments in AI?",
    "published_daterange": {
        "start": "2023-09-17",
        "end": "2021-06-17"
    },
    "domains_allow_list": ["arxiv.org"]
}
```

This isn't just about date ranges. It's about nuanced, tailored searches. In fact, Metaphor Systems has a whole suite of other filters and options that you can use to build a powerful search query. We even use some chain of thought prompting to improve how they use some of these advanced features.

## Case Study 2: Personal Assistant

You ask, "What do I have today?" You want events, emails, reminders. Multiple backends, one unified summary of result. Here you can't even assume that text is going to be embedded in the search backend. You need to model the search backend and the query.

```python
# SearchClient model and OpenAI call
from pydantic import BaseModel
import datetime
import enum
from typing import List
import asyncio

class ClientSource(enum.Enum):
    GMAIL = "gmail"
    CALENDAR = "calendar"

class SearchClient(BaseModel):
    query: str
    keywords: List[str]
    email: str
    source: ClientSource
    start_date: datetime.date
    end_date: datetime.date

    async def execute(self) -> str:
        if self.source == ClientSource.GMAIL:
            return await gmail.search(query=self.query, keywords=self.keywords, email=self.email, start_date=self.start_date, end_date=self.end_date)
        elif self.source == ClientSource.CALENDAR:
            return await calendar.search(query=self.query, keywords=self.keywords, email=self.email, start_date=self.start_date, end_date=self.end_date)

class Retrival(BaseModel):
    queries: List[SearchClient]

    async def execute(self) -> str:
        return await asyncio.gather(*[query.execute() for query in self.queries])

retrival = openai.ChatCompletion.create(
    model="gpt-4",
    response_model=Retrival,
    messages=[
        {"role": "system", "content": "You are Jason's personal assistant."},
        {"role": "user", "content": "What do I have today?"}
    ],
)
```

**Example Output**

```json
{
    "queries": [
        {
            "query": None,
            "keywords": None,
            "email": "jason@example.com",
            "source": "gmail",
            "start_date": "2023-09-17",
            "end_date": None
        },
        {
            "query": None,
            "keywords": ["meeting", "call", "zoom"]]],
            "email": "jason@example.com",
            "source": "calendar",
            "start_date": "2023-09-17",
            "end_date": None

        }
    ]
}
```

`Instructor` models multiple backends and uses multiple dispatch for query preparation. Here you could either use the result to render UI to the client, or ask the chat bot to summarize the results into plain text.

## Conclusion

RAG isn't actually about embeddings. It's about search and query understanding. That's where the magic happens. With `instructor`, you're not just about extraction of structured data, you're building a model of the rest of the world, in this case, a natural language inferface for opinionated search backends.

Building and modeling the system is hard, but `instructor` makes it easier. I'm also working on a new project that will make it even easier to build these systems with a team, collaborating on the schemas and prompts, allowing people to deliberate and discuss the systems, and ultimate finetune models that really nail in the context of your domain. To learn more check out [useinstructor.com](https://useinstructor.com) and please fill out the survey to help me build the best product for you.
