---
draft: False
date: 2024-02-06
slug: langchain
tags:
  - patching
  - open source
authors:
  - synacktraa
---

# Instructor as LangChain `Runnable`

If you want to try this example using `instructor hub`, you can pull it by running

```bash
instructor hub pull --slug langchain --py > langchain_example.py
```

LangChain is a popular framework for developing applications powered by language models. A `Runnable` is a protocol that makes it easy to define custom chains as well as invoke them in a standard way. Utilizing `instructor` library in langchain can help in making better applications. 

By the end of this blog post, you will learn how to effectively utilize Instructor with LangChain.

<!-- more -->

## Patching

Instructor's patch enhances an openai api with the following features:

- `response_model` in `create` calls that returns a pydantic model
- `max_retries` in `create` calls that retries the call if it fails by using a backoff strategy

!!! note "Learn More"

    To learn more, please refer to the [docs](../index.md). To understand the benefits of using Pydantic with Instructor, visit the tips and tricks section of the [why use Pydantic](../why.md) page.

## LangChain Integration

Langchain's `RunnableLambda` class instance is returned with both sync and async create integration.

```python
from typing import Optional, TypeVar, Type

import instructor
from pydantic import BaseModel, Field
from openai import OpenAI, AsyncOpenAI
from langchain.schema.messages import BaseMessage
from langchain.schema.runnable import RunnableLambda
from langchain.adapters.openai import convert_message_to_dict

T = TypeVar('T', bound=BaseModel)


def create_instructor_runnable(
    mode: instructor.Mode,
    model: str,
    response_model: Type[T],
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> RunnableLambda[str | list[BaseMessage], T]: 
    """
    Create an instructor integrated langchain runnable.
    @param mode: instructor mode to use.
    @param model: Model to use.
    @param response_model: pydantic class to format output.
    @param base_url: Base URL to get chat completions.
    @param api_key: API key to use for requesting base URL.
    @param kwargs: Extra kwargs to pass to create method.
    """
    
    client = instructor.patch(
        OpenAI(base_url=base_url, api_key=api_key), mode=mode)
    aclient = instructor.patch(
        AsyncOpenAI(base_url=base_url, api_key=api_key), mode=mode)
    
    load_kwargs = (
        lambda x: {
            'messages':([{'role': 'user', 'content': x}] 
                if isinstance(x, str) 
                else [convert_message_to_dict(msg) for msg in x]),
            'model': model,
            'response_model': response_model,
        } | kwargs 
    )

    def func(__arg):
        return client.chat.completions.create(**load_kwargs(__arg))
    
    async def afunc(__arg):
        return await aclient.chat.completions.create(**load_kwargs(__arg))

    return RunnableLambda(func=func, afunc=afunc)


class Character(BaseModel):
    name: str
    fact: str = Field(..., description="A fact about the character")

"""Create Ollama instructor runnable 
(leave base_url empty & update model and api_key parameter if you're using openai)"""
llm = create_instructor_runnable(
    mode=instructor.Mode.JSON,
    model='llama2',
    response_model=Character, 
    base_url="http://localhost:11434/v1", # Ollama default URL
    api_key="ollama" # not required, you can leave it empty
)

print(llm.invoke('Tell me about Elon Musk.'))
# Character(name='Elon Musk', fact='He is the CEO of SpaceX and Tesla, Inc.')

import asyncio
from langchain.schema.messages import SystemMessage, HumanMessage
print(asyncio.run(llm.ainvoke([
    SystemMessage(content='Answer query related to anime "Bleach"'),
    HumanMessage(content='Who is Ichigo Kurosaki.')])))
# Character(name='Ichigo Kurosaki', fact='He can transform into a Shinigami and play the drum.')

for character in llm.batch([
    "I go by the name SynAcktra. I like learning about AI.",
    "Jason Liu is the creator of instructor library."
]):
    print(character)
# Character(name='SynAcktra', fact='Learning about AI is his passion.')
# Character(name='Jason Liu', fact='Jason Liu is the creator of Instructor Library.')
```