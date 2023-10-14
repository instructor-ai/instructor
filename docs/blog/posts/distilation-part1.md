---
draft: False 
date: 2023-10-17
tags:
  - RAG
  - Finetuning
---

# Introduction to `Instructions` from `Instructor`, finetuning from Python functions.

The core philosophy with the `instructor` library is to make language models backwards compatible with existing code. By adding Pydantic in the mix we're able to easily work with LLMs without much worry.

However, many times, a single function isn't just one LLM call. After the results are returned theres [validation](/docs/validation.md), some additional processing and formatting before you `return` the result.

But the promise of LLMs is that they can do all of this in one go. So how do we get there? Finetuning end to end is a great tool for enhancing language models. Instructor uses type hints via Pydantic to maintain backward compatibility. Distillation focuses on fine-tuning language models to imitate specific functions.

## Challenges in Fine-tuning

Fine-tuning a model isn't as straightforward as just writing `def f(a, b): return a * b` to teach a model three-digit multiplication. Substantial data preparation is required, making logging for data collection cumbersome. Luckily OpenAI not only provides a fine-tuning script but also one for function calling which simplies the process backed to structured outputs! More over, the finetune allows us to avoid passing the schema to the model, resulting in less tokens being used!

## Role of Instructor in Easing the Process

The feature `from instructor import Instructions` simplifies this. It decorates Python functions that return Pydantic objects, automatically creating a fine-tuning dataset when provided a handler for logging. This allows you to finetune a model to imitate a function's behavior.

## How to Use Instructor's Distillation Feature

Here's an example to illustrate its use:

```python
import logging
from pydantic import BaseModel
from instructor import Instructions

logging.basicConfig(level=logging.INFO)

instructions = Instructions(
    name="three_digit_multiply",
    finetune_format="messages",
    log_handlers=[logging.FileHandler("math_finetunes.jsonl")]
)

class Multiply(BaseModel):
    a: int
    b: int
    result: int

@instructions.distil
def fn(a: int, b: int) -> Multiply:
    resp = a * b
    return Multiply(a=a, b=b, result=resp)

for _ in range(10):
    a = random.randint(100, 999)
    b = random.randint(100, 999)
    print(fn(a, b))
```

## Logging output

```python
{
    "messages": [
        {"role": "system", "content": 'Predict the results of this function: ...'},
        {"role": "user", "content": 'Return fn(133, b=539)'},
        {"role": "assistant", 
            "function_call": 
                {
                    "name": "Multiply", 
                    "arguments": '{"a":133,"b":539,"result":89509}'
            }
        }
    ],
    "functions": [
        {"name": "Multiply", "description": "Correctly extracted `Multiply`..."}
    ]
}
```

## Why Instructor and Distillation are Useful

Many systems are not as simple as a single `openai.ChatCompletion.create` call, instead we often create objects, do additional processing, validation, error correction, and then return the result. This is a lot of work, and it's easy to make mistakes. Instructor's `distil` feature makes this process easier by:

1. Streamlines complex functions with validations, making them more efficient.
2. Facilitates the integration of classical machine learning with language models. 

By understanding and leveraging these capabilities, you can create powerful, fine-tuned language models with ease. To learn more about how to use the file to finetune a model, check out the [cli](/docs/cli/finetune.md)

## Next Steps

This post is mostly a peek of what I've been working on this week. Once we have a model trained I'd like to be able to dynamically swap the implemetnation of a function with a model. This would allow us to do things like:

```python
from instructor import Instructions

instructions = Instructions(
    name="three_digit_multiply",
)

@instructions.distil(model='gpt-3.5-turbo:finetuned', swap=True)
def fn(a: int, b: int) -> Multiply:
    resp = a + b
    return Multiply(a=a, b=b, result=resp)
```

Now we can swap out the implementation of `fn` with calling the finetuned model, since we know the response type is still `Multiply` we can use instructor behind the scenes and have it be backwards compatible with the existing code. 

This is a powerful idea, and I'm excited to see where it goes.