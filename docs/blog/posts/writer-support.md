---
authors:
  - ivanleomk
  - yanomaly
categories:
  - Writer SDK
comments: true
date: 2024-11-19
description: Announcing Writer integration with Instructor for structured outputs and enterprise AI workflows
draft: false
slug: writer-support
tags:
  - Writer
  - Enterprise AI
  - Integrations
---

# Structured Outputs with Writer now supported

We're excited to announce that `instructor` now supports [Writer](https://writer.com)'s enterprise-grade LLMs, including their latest Palmyra X 004 model. This integration enables structured outputs and enterprise AI workflows with Writer's powerful language models.

## Getting Started

First, install `instructor` with Writer support by running `pip install instructor[writer]` in your terminal. Then, head over to [Writer](https://writer.com) to sign up for Writer access and get an API key. You can paste API key as environment variable _WRITER_API_KEY_, or set attribute for Writer constructor.

<!-- more -->

```python
import instructor
from writerai import Writer
from pydantic import BaseModel

# Initialize Writer client
client = instructor.from_writer(Writer(api_key="your API key"))


class User(BaseModel):
    name: str
    age: int


# Extract structured data
user = client.chat.completions.create(
    model="palmyra-x-004",
    messages=[{"role": "user", "content": "Extract: John is 30 years old"}],
    response_model=User,
)

print(user)
#> name='John' age=30
```

!!! note

    If you'd like to use the Async version of the Writer client, you can do so by using `instructor.from_writer(AsyncWriter())`.

We also support streaming with the Writer client using our `create_partial` method. This allows you to process responses incrementally as they arrive.

This is particularly valuable for maintaining responsive applications and delivering a smooth user experience, especially when dealing with larger responses so that users can see immediate results.

```python
import instructor
from writerai import Writer
from pydantic import BaseModel

# Initialize Writer client
client = instructor.from_writer(Writer())


text_block = """
In our recent online meeting, participants from various backgrounds joined to discuss the upcoming tech conference. The names and contact details of the participants were as follows:

- Name: John Doe, Email: johndoe@email.com, Twitter: @TechGuru44
- Name: Jane Smith, Email: janesmith@email.com, Twitter: @DigitalDiva88
- Name: Alex Johnson, Email: alexj@email.com, Twitter: @CodeMaster2023

During the meeting, we agreed on several key points. The conference will be held on March 15th, 2024, at the Grand Tech Arena located at 4521 Innovation Drive. Dr. Emily Johnson, a renowned AI researcher, will be our keynote speaker.

The budget for the event is set at $50,000, covering venue costs, speaker fees, and promotional activities. Each participant is expected to contribute an article to the conference blog by February 20th.

A follow-up meetingis scheduled for January 25th at 3 PM GMT to finalize the agenda and confirm the list of speakers.
"""


class User(BaseModel):
    name: str
    email: str
    twitter: str


class MeetingInfo(BaseModel):
    date: str
    location: str
    budget: int
    deadline: str


PartialMeetingInfo = instructor.Partial[MeetingInfo]


extraction_stream = client.chat.completions.create(
    model="palmyra-x-004",
    messages=[
        {
            "role": "user",
            "content": f"Get the information about the meeting and the users {text_block}",
        },
    ],
    response_model=PartialMeetingInfo,
    stream=True,
)  # type: ignore


for obj in extraction_stream:
    print(obj)
    #> date='March 15th, 2024' location='' budget=None deadline=None
    #> date='March 15th, 2024' location='Grand Tech Arena, 4521 Innovation' budget=None deadline=None
    #> date='March 15th, 2024' location='Grand Tech Arena, 4521 Innovation Drive' budget=50000 eadline='February 20th'
```

As well, if you don't sure that your request will be successfully performed from the first attempt, don't excite we also support retries mechanism! It's quite easy in use:

```python
import instructor
from typing import Annotated
from writerai import Writer
from pydantic import BaseModel, AfterValidator, Field

#Initialize Writer client
client = instructor.from_writer(Writer())


#Example of model, that may require usage of retries
def uppercase_validator(v):
    if v.islower():
        raise ValueError("Name must be in uppercase")
    return v


class User(BaseModel):
    name: Annotated[str, AfterValidator(uppercase_validator)] = Field(
        ..., description="The name of the user"
    )
    age: int
    

user = client.chat.completions.create(
    model="palmyra-x-004",
    messages=[{"role": "user", "content": "Extract: jason is 12"}],
    response_model=User,
    max_retries=3,
)

print(user)
#> name='JASON' age=12
```

And that’s it! We showed only small part of things you can do featuring Writer provider: different types of classification, sentimental analysis and other cool stuff are ready to be used out of the box. We're excited to see what you build with Instructor and Writer! If you have any questions about Writer, do check out the [Writer documentation](https://dev.writer.com/home/introduction).
