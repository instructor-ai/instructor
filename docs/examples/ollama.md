# Running a Local Ollama Model

Here are some instructions on using Ollamo and Litellm.

## Instructions

1. Install Ollama by visiting the website [https://ollama.ai/download](https://ollama.ai/download) and selecting the appropriate operating system.

2. Once installed, open the Ollama app, which should be running in your taskbar.

3. Open the terminal and download a model. For example, to download the llama2 model, run the command:

```bash
ollama run llama2
```

4. In your terminal, start your virtual environment and install the 'litellm[proxy]' package using poetry you can run the command:

```bash
pip install 'litellm[proxy]'
```

Then you should be able to patch using the wrap completion API.
Since it's just going to use regular prompting and not... Function Calling. You'll need to have a lot more instructions in the system message to ask it to output JSON.

```python
from litellm import completion
from pydantic import BaseModel

import instructor
from instructor.patch import wrap_chatcompletion

completion = wrap_chatcompletion(completion, mode=instructor.Mode.MD_JSON)


class UserExtract(BaseModel):
    name: str
    age: int


user = completion(
    model="ollama/llama2",
    response_model=UserExtract,
    messages=[
        {
            "role": "system",
            "content": "You are a JSON extractor. Please extract the following JSON, No Talk.",
        },
        {
            "role": "user",
            "content": "Extract `My name is Jason and I am 25 years old`",
        },
    ],
)

assert isinstance(user, UserExtract), "Should be instance of UserExtract"
assert user.name.lower() == "jason"
assert user.age == 25
```
