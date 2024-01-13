# Field Level Streaming

Field level streaming provides incremental snapshots of the current state of the response model that are immediately useable. This approach is particularly relevant in contexts like rendering UI components.

Instructor supports this pattern by making use of `Partial[T]`. This lets us dynamically create a new class that treats all of the original model's fields as `Optional`.

When specifying a partial response model and setting streaming to true, the response from Instructor becomes a generator. As the generator yields results, you can iterate over these incremental updates. The last value yielded by the generator represents the completed extraction.

!!! warning "Limited Validator Support"

    Important: Fewer validators are supported by `Partial` response models as streamed fields will natural raise validation errors

!!! note "Item Level Streaming"

    If you are looking for wider validator support or to stream out a list of completed objects one by one, take a look at [Multi-task Streaming](./lists.md).

Lets look at an example of streaming an extraction of conference information.

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel
from typing import List

client = instructor.patch(OpenAI())

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
    users: List[User]
    date: str
    location: str
    budget: int
    deadline: str


PartialMeetingInfo = instructor.Partial[MeetingInfo]


extraction_stream = client.chat.completions.create(
    model="gpt-4",
    response_model=PartialMeetingInfo,
    messages=[
        {
            "role": "user",
            "content": f"Get the information about the meeting and the users {text_block}",
        },
    ],
    stream=True,
)  # type: ignore


from rich.console import Console

console = Console()

for extraction in extraction_stream:
    obj = extraction.model_dump()
    console.clear()
    console.print(obj)

```

![Partial Streaming Gif](../img/partial_streaming.gif)
