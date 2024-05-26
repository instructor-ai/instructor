---
draft: False
date: 2024-05-26
slug: reka
tags:
  - patching
authors:
  - TootyFrooties
---

# Structured Outputs using Reka
You can now also use Reka models for inference by using from_reka.

If you want to try this example using `instructor hub`, you can pull it by running

```bash
instructor hub pull --slug reka --py > reka_example.py
```



## Reka API
The examples are using reka core. For more detailed Reka documentation visit [Reka docs](https://docs.reka.ai/index.html)

To use Reka you need to obtain a Reka API key.
Goto [Reka AI](https://reka.ai/) click on API Access and login. Select API Keys from the left menu and then select 
Create API key to create a new key. You need to fund your account before use.

Currently Reka does not support async

## Use example
Some pip packages need to be installed to use the example:
```
pip install instructor reka-api pydantic

```

An example:
```python
import os
from pydantic import BaseModel, Field
from typing import List
import reka
from instructor import from_reka, Mode

class UserDetails(BaseModel):
    name: str
    age: int


# enables `response_model` in chat call
client = from_reka(api_key=os.environ.get("REKA_API_KEY"))

user_info = client.chat.completions.create(
    response_model=UserDetails,
    messages=[{"role": "user", "content": "Extract John Doe is 30 years old."}],
    temperature=0,
)

print(user_info.name)
#> John Doe
print(user_info.age)
#> 30