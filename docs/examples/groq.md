# Structured Outputs using Groq
Instead of using openai or antrophic you can now also use groq for inference by using from_groq.

The examples are using mixtral-8x7b model.

## GroqCloud API
To use groq you need to obtain a groq API key.
Goto [groqcloud](https://console.groq.com) and login. Select API Keys from the left menu and then select Create API key to create a new key.

## Use example
Some pip packages need to be installed to use the example:
```
pip install instructor groq pydantic openai anthropic
```
You need to export the groq API key:
```
export GROQ_API_KEY=<your-api-key>
```

An example:
```python
import os
from pydantic import BaseModel, Field
from typing import List
from groq import Groq
import instructor

class Character(BaseModel):
    name: str
    fact: List[str] = Field(..., description="A list of facts about the subject")


client = Groq(
    api_key=os.environ.get('GROQ_API_KEY'),
)

client = instructor.from_groq(client, mode=instructor.Mode.TOOLS)

resp = client.chat.completions.create(
    model="mixtral-8x7b-32768",
    messages=[
        {
            "role": "user",
            "content": "Tell me about the company Tesla",
        }
    ],
    response_model=Character,
)
print(resp.model_dump_json(indent=2))
"""
{
  "name": "Tesla",
  "fact": [
    "An American electric vehicle and clean energy company.",
    "Co-founded by Elon Musk, JB Straubel, Martin Eberhard, Marc Tarpenning, and Ian Wright in 2003.",
    "Headquartered in Austin, Texas.",
    "Produces electric vehicles, energy storage solutions, and more recently, solar energy products.",
    "Known for its premium electric vehicles, such as the Model S, Model 3, Model X, and Model Y.",
    "One of the world's most valuable car manufacturers by market capitalization.",
    "Tesla's CEO, Elon Musk, is also the CEO of SpaceX, Neuralink, and The Boring Company.",
    "Tesla operates the world's largest global network of electric vehicle supercharging stations.",
    "The company aims to accelerate the world's transition to sustainable transport and energy through innovative technologies and products."
  ]
}
"""
```
You can find another example called groq_example2.py under examples/groq of this repository.