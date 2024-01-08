# Streaming with Instructor and Ollama

Streaming responses from a model can be a game-changer for real-time applications. This example demonstrates how Instructor, in conjunction with Ollama, can be leveraged to stream structured data extracted from a language model. Let's delve into the nuances of this implementation and understand how Instructor facilitates this process.

## The Power of Streaming

Streaming is a technique that allows us to process data in real-time as it's being generated. This is particularly useful when dealing with large volumes of data or when we want to display information to the user as soon as it's available, without waiting for the entire response. This is especially useful in real-time applications where we want to display information to the user as soon as it's available, without waiting for the entire response.

First we need to get Ollama running on our machine. Follow the setup [here](https://github.com/jmorganca/ollama)if you have not done so already. 

For this example we use OpenHermes v2.5. To download run this command in your terminal
```bash
ollama pull openhermes:v2.5
```

```python
from litellm import completion
from pydantic import BaseModel
from typing import Iterable


import instructor
from instructor.patch import wrap_chatcompletion

completion = wrap_chatcompletion(completion, mode=instructor.Mode.MD_JSON)
```

In the snippet above, we wrap the `completion` function from `litellm` with Instructor's `wrap_chatcompletion`. This enables us to use Instructor's mode for Markdown and JSON, which is essential for structuring the output of our language model.

## Defining the Response Model

Just like in `models.md`, we define our response model using Pydantic's `BaseModel`. This model will dictate the structure of the data we expect to receive from the language model.

```python

class User(BaseModel):
    name: str
    age: int

Users = Iterable[User]
```

Here, `User` is a simple model with `name` and `age` fields. We also define `Users` as an iterable of `User` instances, which will be our expected output format.

## Crafting the Prompt

Following the principles outlined in `prompting.md`, we craft a prompt that is self-descriptive and modular. The prompt is designed to instruct the language model to segment a given dataset into entities and ensure the output is valid JSON.

```python
dataset = [
        "My name is John and I am 20 years old",
        "My name is Mary and I am 21 years old",
        "My name is Bob and I am 22 years old",
        "My name is Alice and I am 23 years old",
        "My name is Jane and I am 24 years old",
        "My name is Joe and I am 25 years old",
        "My name is Jill and I am 26 years old",
    ]

users = completion(
    model="ollama/openhermes:v2.5",
    response_model=Users,
    messages=[
        {
            "role": "system",
            "content": "You are a JSON Output system, only return valid JSON. YOU CAN ONLY RETURN WITH JSON NO TALKING",
        },
        {
            "role": "user",
            "content": f"Consider the data below:\n{dataset}\n"
                "Correctly segment it into entitites\n"
                "Make sure the JSON is correct",
        },
    ],
    api_base="http://localhost:11434",
    stream=True,
    max_tokens=100,
)
```

The `completion` function is called with the `model`, `response_model`, and `messages` parameters. The `stream=True` parameter is crucial as it indicates that we want to receive the data as a stream.

## Streaming the Response

Finally, we iterate over the streamed response and print out each `User` instance. This loop will execute as soon as each piece of data is available, showcasing the real-time capabilities of streaming.

```python
for user in users:
    print(user)
```

![Streaming with Instructor and Ollama](./docs/concepts/mistral-ollama-streaming-demo.gif)

In this example, we've seen how Instructor can be used to facilitate streaming responses from local models trhough Ollama. By defining a clear response model and crafting a precise prompt, we can extract structured data from the language model's output and stream it to the user in real-time. This approach is invaluable for applications that require immediate data processing and display ensuring a seamless and responsive user experience.
