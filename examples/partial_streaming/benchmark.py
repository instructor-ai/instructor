# Part of this code is adapted from the following examples from OpenAI Cookbook:
# https://cookbook.openai.com/examples/how_to_stream_completions
# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
import time
import tiktoken
import instructor
from openai import OpenAI
from pydantic import BaseModel
from typing import List


def num_tokens_from_string(string: str, model_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model_name)

    num_tokens = len(encoding.encode(string))
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>

    return num_tokens


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
schema = instructor.openai_schema(PartialMeetingInfo).openai_schema

def benchmark_response(response_model=None, model="gpt-4"):
    client = instructor.patch(OpenAI())
    
    if response_model:
        content = f"Get the information about the meeting and the users {text_block}"
    else:
        content = f"Get the information about the meeting and the users {text_block}. Respond only in JSON that adheres to this schema: {schema} otherwise something bad will happen"

    start_time = time.time()
    extraction_stream = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": content
            }
        ],
        response_model=response_model,
        stream=True,
    )

    collected_messages = []
    for extraction in extraction_stream:
        chunk_time = time.time() - start_time
        print(extraction._raw_response)
        return
        if response_model:
            collected_messages.append(extraction)
        else:
            collected_messages.append(extraction.choices[0].delta.content)

    collected_messages = [m for m in collected_messages if m is not None]
    collected_messages = "".join(collected_messages)
    print(collected_messages)
    print(PartialMeetingInfo.model_validate_json(collected_messages))
    print(f"Full response received {chunk_time:.2f} seconds after request")

    # full_reply_content = "".join([m for m in collected_messages if m is not None])

    # output_tokens = num_tokens_from_string(full_reply_content, model)
    # char_per_sec = output_tokens / chunk_time

    # print(f"{output_tokens} total output tokens")
    # print(f"{char_per_sec:.2f} tok/s")


# benchmark_response()
benchmark_response(PartialMeetingInfo)
# benchmark_response(instructor.function_calls.Mode.MD_JSON)
# print()
# benchmark_response(instructor.function_calls.Mode.FUNCTIONS)

