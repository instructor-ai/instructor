---
draft: False
date: 2024-03-07
slug: open-source-local-structured-output-pydantic-json-openai
tags:
  - llms
  - opensource
  - together
  - llama-cpp-python
  - anyscale
  - groq
  - mistral
  - ollama
authors:
  - jxnl
---

# Structured Output for Open Source and Local LLMS 

Originally, Instructor facilitated API interactions solely via the OpenAI SDK, with an emphasis on function call by incorporating [Pydantic](https://pydantic-docs.helpmanual.io/) for structured data validation and serialization. 


As the year progressed, we expanded our toolkit by integrating [JSON mode](../../concepts/patching.md#json-mode), thus enhancing our adaptability to vision models and open source models. This advancement now enables us to support an extensive range of models, from [GPT](https://openai.com/api/) and [Mistral](https://mistral.ai) to virtually any model accessible through [Ollama](https://ollama.ai) and [Hugging Face](https://huggingface.co/models), facilitated by [llama-cpp-python](../../hub/llama-cpp-python.md). For more insights into leveraging JSON mode with various models, refer back to our detailed guide on [Patching](../../concepts/patching.md).

If you want to check out a course on how to use Instructor with Pydantic, check out our course on [Steering language models towards structured outputs.](https://www.wandb.courses/courses/steering-language-models).

<!-- more -->


## Exploring Different OpenAI Clients with Instructor

The landscape of OpenAI clients is diverse, each offering unique functionalities tailored to different needs. Below, we explore some of the notable clients integrated with Instructor, providing structured outputs and enhanced capabilities, complete with examples of how to initialize and patch each client.

## Local Models

### Ollama: A New Frontier for Local Models

Ollama's introduction significantly impacts the open-source community, offering a way to merge structured outputs with local models via JSON schema, as detailed in our [Ollama documentation](../../hub/ollama.md).

For an in-depth exploration of Ollama, including setup and advanced features, refer to the documentation. The [Ollama official website](https://ollama.ai/download) also provides essential resources, model downloads, and community support for newcomers.

```
ollama run llama2
```

```python
from openai import OpenAI
from pydantic import BaseModel
import instructor


class UserDetail(BaseModel):
    name: str
    age: int


# enables `response_model` in create call
client = instructor.from_openai(
    OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # required, but unused
    ),
    mode=instructor.Mode.JSON,
)

user = client.chat.completions.create(
    model="llama2",
    messages=[
        {
            "role": "user",
            "content": "Jason is 30 years old",
        }
    ],
    response_model=UserDetail,
)

print(user)
#> name='Jason' age=30
```

### llama-cpp-python

Open-source LLMS are gaining popularity, and llama-cpp-python has made the `llama-cpp` model available to obtain structured outputs using JSON schema via a mixture of [constrained sampling](https://llama-cpp-python.readthedocs.io/en/latest/#json-schema-mode) and [speculative decoding](https://llama-cpp-python.readthedocs.io/en/latest/#speculative-decoding). They also support a [OpenAI compatible client](https://llama-cpp-python.readthedocs.io/en/latest/#openai-compatible-web-server), which can be used to obtain structured output as an in-process mechanism to avoid any network dependency.

For those interested in leveraging the power of llama-cpp-python for structured outputs, here's a quick example:


```python
import llama_cpp
import instructor

from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
from pydantic import BaseModel


llama = llama_cpp.Llama(
    model_path="../../models/OpenHermes-2.5-Mistral-7B-GGUF/openhermes-2.5-mistral-7b.Q4_K_M.gguf",
    n_gpu_layers=-1,
    chat_format="chatml",
    n_ctx=2048,
    draft_model=LlamaPromptLookupDecoding(num_pred_tokens=2),
    logits_all=True,
    verbose=False,
)


create = instructor.patch(
    create=llama.create_chat_completion_openai_v1,
    mode=instructor.Mode.JSON_SCHEMA, 
)

class UserDetail(BaseModel):
    name: str
    age: int


user = create(
    messages=[
        {
            "role": "user",
            "content": "Extract `Jason is 30 years old`",
        }
    ],
    response_model=UserDetail,
)

print(user)
#> name='Jason' age=30
```

## Alternative Providers

### Anyscale

Anyscale's Mistral model, as detailed in our [Anyscale documentation](../../hub/anyscale.md) and on [Anyscale's official documentation](https://docs.anyscale.com/), introduces the ability to obtain structured outputs using JSON schema.

```bash
export ANYSCALE_API_KEY="your-api-key"
```

```python
import os
from openai import OpenAI
from pydantic import BaseModel
import instructor


class UserDetails(BaseModel):
    name: str
    age: int


# enables `response_model` in create call
client = instructor.from_openai(
    OpenAI(
        base_url="https://api.endpoints.anyscale.com/v1",
        api_key=os.environ["ANYSCALE_API_KEY"],
    ),
    # This uses Anyscale's json schema output mode
    mode=instructor.Mode.JSON_SCHEMA,
)

resp = client.chat.completions.create(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    messages=[
        {"role": "system", "content": "You are a world class extractor"},
        {"role": "user", "content": 'Extract the following entities: "Jason is 20"'},
    ],
    response_model=UserDetails,
)
print(resp)
#> name='Jason' age=20
```

### Groq

Groq's platform, detailed further in our [Groq documentation](../../hub/groq.md) and on [Groq's official documentation](https://groq.com/), offers a unique approach to processing with its tensor architecture. This innovation significantly enhances the performance of structured output processing.

```bash
export GROQ_API_KEY="your-api-key"
```

```python
import os
import instructor
import groq
from pydantic import BaseModel

client = qrog.Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# By default, the patch function will patch the ChatCompletion.create and ChatCompletion.create methods to support the response_model parameter
client = instructor.from_openai(client, mode=instructor.Mode.MD_JSON)


# Now, we can use the response_model parameter using only a base model
# rather than having to use the OpenAISchema class
class UserExtract(BaseModel):
    name: str
    age: int


user: UserExtract = client.chat.completions.create(
    model="mixtral-8x7b-32768",
    response_model=UserExtract,
    messages=[
        {"role": "user", "content": "Extract jason is 25 years old"},
    ],
)

assert isinstance(user, UserExtract), "Should be instance of UserExtract"
print(user)
#> name='jason' age=25
"""
```

### Together AI

Together AI, when combined with Instructor, offers a seamless experience for developers looking to leverage structured outputs in their applications. For more details, refer to our [Together AI documentation](../hub/together.md) and explore the [patching guide](../concepts/patching.md) to enhance your applications.

```bash
export TOGETHER_API_KEY="your-api-key"
```

```python
import os
import openai
from pydantic import BaseModel
import instructor

client = openai.OpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=os.environ["TOGETHER_API_KEY"],
)

client = instructor.from_openai(client, mode=instructor.Mode.TOOLS)

class UserExtract(BaseModel):
    name: str
    age: int


user: UserExtract = client.chat.completions.create(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    response_model=UserExtract,
    messages=[
        {"role": "user", "content": "Extract jason is 25 years old"},
    ],
)

assert isinstance(user, UserExtract), "Should be instance of UserExtract"
print(user)

#> name='jason' age=25
```

### Mistral

For those interested in exploring the capabilities of Mistral Large with Instructor, we highly recommend checking out our comprehensive guide on [Mistral Large](../../hub/mistral.md).

```python
import instructor

from pydantic import BaseModel
from mistralai.client import MistralClient

client = MistralClient()

patched_chat = instructor.from_openai(create=client.chat, mode=instructor.Mode.MISTRAL_TOOLS)

class UserDetails(BaseModel):
    name: str
    age: int

resp = patched_chat(
    model="mistral-large-latest",
    response_model=UserDetails,
    messages=[
        {
            "role": "user",
            "content": f'Extract the following entities: "Jason is 20"',
        },
    ],
)
print(resp)
#> name='Jason' age=20
```
