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

# RAG is more than just embedding search

With the advent of large language models (LLM), retrival augmented generation (RAG) has become a hot topic. However throught the past year of [helping startups](https://jxnl.notion.site/Working-with-me-ec2bb36a5ac048c2a8f6bd888faea6c2?pvs=4) integrate LLMs into their stack I've noticed that the pattern of taking user queries directly into LLMs is effectively demoware.

!!! note "What is RAG?"
    Retrival augmented generation (RAG) is a technique that uses a LLM to generate responses, but uses a search backend to augment the generation, in the past year using text embeddings with a vector databases has been the most popular approach. 

<figure markdown>
  ![RAG](img/dumb_rag.png)
  <figcaption>Simple RAG that embedded the user query and makes a search.</figcaption>
</figure>

In this post I'll show you how to use `instructor` to model a search backend, why that might be useful, and how to use it to completely bring your own arbitrary infrastructure to the table.

## The 'Dumb' RAG Model

When you ask a question like, "what is the capital of France?" The RAG 'dumb' model embeds the query and searches in some unopinonated search endpoint. Limited to a single method API like `search(query: str) -> List[str]`. This is fine for simple queries, since you'd expect words like 'paris is the capital of france' to be in the top results of say, your wikipedia embeddings.

However it'd be crazy to think that this was the best solution, its common to also want to filter keywords, or tags, or search in a specific date range, or dispatch to multiple backends where data is located, call other external apis and etc.

This is where function calling comes in, and more interestingly, modeling your function as multiple dispatched functions.

## Improving the RAG Model with Query Understanding

!!! note "Shoutouts"
    Much of this work has been inspired by / done in collab with a few of my clients at [new.computer](https://new.computer), [Metaphor Systems](https://metaphor.systems),  and [Naro](https://narohq.com), go check them out!


Ultimately what you want to deploy is a [system that understands](https://en.wikipedia.org/wiki/Query_understanding) how to take the query and rewrite it to improve precision and recall. Once you have that in place, then the role of the LLM is both translate the user query into a search engine call, render the results, or use the model to summarize the results back ot the user in a coherent way.

<figure markdown>
  ![RAG](img/query_understanding.png)
  <figcaption>Query Understanding system routes to multiple search backends.</figcaption>
</figure>


## Case Study 1: Metaphor Systems

Take [Metaphor Systems](https://metaphor.systems), which turns natural language queries into their custom search-optimized query. If you take a look web ui you'll notice that they have an auto-prompt option, which uses function calls to furthur optimize your query using an language model, and turn it into a fully specified metaphor systems query.

<figure markdown>
![Metaphor Systems](img/meta.png)
<figcaption>Metaphor Systems UI</figcaption>
</figure>

If we peek under the hood, we can see that the query is actually a complex object, with a date range, and a list of domains to search in. Its actually more complex than this but this is a good start.

```python
from pydantic import BaseModel
from typing import List

import datetime
import instructor 
import openai

# Enables response_model in the openai client
instructor.patch()

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

This isn't just about adding some date ranges. It's about nuanced, tailored searches, that is deeply integrated with the backend. Metaphor Systems has a whole suite of other filters and options that you can use to build a powerful search query. They can even use some chain of thought prompting to improve how they use some of these advanced features.

```python
class DateRange(BaseModel):
    start: datetime.date
    end: datetime.date
    chain_of_thought: str = Field(
        None,
        description="Think step by step to plan what is the best time range to search in"
    )
```

## Case Study 2: Personal Assistant

Another great example of this multiple dispatch pattern is a personal assistant. You ask, "What do I have today?" You want events, emails, reminders. Multiple backends, one unified summary of result. Here you can't even assume that text is going to be embedded in the search backend. You need to model the search backend and the query.

```python
# SearchClient model and OpenAI call
from pydantic import BaseModel
from typing import List

import datetime
import enum
import asyncio
import instructor 
import openai

# Enables response_model in the openai client
instructor.patch()

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

Notice that we have a list of queries, that route to different search backends, email and calendar. We can even dispatch them async to be as performance as possible. Not only do we dispatch to different backends (that we have no control over), but you are likely going to render them to the user differently as well, perhaps you want to summarize the emails in text, but you want to render the calendar events as a list that they can scroll across on a mobile app.

Both of these examples show case how both search providors and consumers can use `instructor` to model their systems. This is a powerful pattern that allows you to build a system that can be used by anyone, and can be used to build a LLM layer, from scratch, in front of any arbitrary backend.

## Conclusion

At the end of the day its not really about embedding systems, its about good old information retrival, query understanding, rewriting, etc. That's where the magic happens. With `instructor`, you're not just working with extraction of structured data, you're building a model of a system and presenting it to the language model. It super charges function calling to reason about multiple calls to many systes in a pythonic way.

## What's Next?

The real game-changer isn't just smarter algorithms; it's the collaboration between expert users and AI engineers. Experts bring domain-specific insights, but how do you distill this wisdom into tool use? That's the gap I'm trying to tackle next. I'm building a platform that fosters collaboration between the two, making it easier to fine-tune prompts or even the language models themselves real-world expertise. Intrigued? Visit [useinstructor.com](https://useinstructor.com) and take our survey. Together, let's create tools that are as brilliant as the minds using them.