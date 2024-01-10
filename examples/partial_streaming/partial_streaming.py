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

A follow-up meeting is scheduled for January 25th at 3 PM GMT to finalize the agenda and confirm the list of speakers.
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
        model="gpt-3.5-turbo-0613",
        response_model=PartialMeetingInfo,
        messages=[
            {
                "role": "user",
                "content": f"Get only the user details for this information about the user {text_block}",
            },
        ],
        stream=True,
    )  # type: ignore


for extraction in extraction_stream:
    print(extraction)

print("extraction complete")
print(extraction)