# Distilling python functions into LLM

`Instructions` from the `Instructor` library offers a seamless way to make language models backward compatible with existing Python functions. By employing Pydantic type hints, it not only ensures compatibility but also facilitates fine-tuning language models to emulate these functions end-to-end.

## The Challenges in Function-Level Fine-Tuning

Unlike simple script-level fine-tuning, replicating the behavior of a Python function in a language model involves intricate data preparation. For instance, teaching a model to execute three-digit multiplication is not as trivial as implementing `def f(a, b): return a * b`. OpenAI's fine-tuning script coupled with their function calling utility provides a structured output, thereby simplifying the data collection process. Additionally, this eliminates the need for passing the schema to the model, thus conserving tokens.

## The Role of `Instructions` in Simplifying the Fine-Tuning Process

By using `Instructions`, you can annotate a Python function that returns a Pydantic object, thereby automating the dataset creation for fine-tuning. A handler for logging is all that's needed to build this dataset.

## How to Implement `Instructions` in Your Code

Here's a step-by-step example:

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

class Response(BaseModel):
    a: int
    b: int
    result: int

@instructions.distil
def fn(a: int, b: int) -> Response:
    resp = a + b
    return Response(a=a, b=b, result=resp)
```

## Custom Log Handlers for Data Collection

While the example above uses a file-based log handler, you can easily extend this to custom log handlers for different storage solutions. The following skeleton code illustrates how to create a log handler for an S3 bucket:

```python
import logging
import boto3

class S3LogHandler(logging.Handler):
    def __init__(self, bucket, key):
        logging.Handler.__init__(self)
        self.bucket = bucket
        self.key = key

    def emit(self, record):
        s3 = boto3.client('s3')
        log_entry = self.format(record)
        s3.put_object(Body=log_entry, Bucket=self.bucket, Key=self.key)
```

You can add this custom log handler to `Instructions` as shown:

```python
instructions = Instructions(
    name="three_digit_multiply",
    finetune_format="messages",
    log_handlers=[S3LogHandler(bucket='your-bucket', key='your-key')]
)
```

## Why `Instructions` is a Game-Changer

1. It condenses complex, multi-step functions with validations into a single fine-tuned model.
2. It integrates language models with classical machine learning seamlessly.

## Next Steps and Future Scope

Going forward, the aim is to dynamically switch between the Python function and its fine-tuned model representation. This could look like:

```python
from instructor import Instructions

instructions = Instructions(
    name="three_digit_multiply",
)

@instructions.distil(model='gpt-3.5-turbo:finetuned', swap=True)
def fn(a: int, b: int) -> Response:
    resp = a + b
    return Response(a=a, b=b, result=resp)
```

This dynamic switching retains backward compatibility while improving efficiency, opening up exciting avenues for future developments.