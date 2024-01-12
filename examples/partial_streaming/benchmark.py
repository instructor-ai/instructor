# Part of this code is adapted from the following examples from OpenAI Cookbook:
# https://cookbook.openai.com/examples/how_to_stream_completions
# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
import time
import tiktoken
import instructor
from openai import OpenAI
from pydantic import BaseModel
from typing import List

"""
Comparing the total time and tok/s between streaming raw tokens and validating the final result against a pydantic model
versus using partial streaming from instructor
"""

client = instructor.patch(OpenAI())

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

def benchmark_raw_stream(model="gpt-4"):
    print("Raw Streaming Benchmark Start")

    content = f"""Get the information about the meeting and the users {text_block}. 
    Respond only in JSON tha would validate to this schema and include nothing extra. 
    Otherwise something bad will happen:\n {PartialMeetingInfo.model_json_schema()}"""

    start_time = time.time()
    extraction_stream = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": content
            }
        ],
        stream=True,
    )

    collected_messages = []
    for extraction in extraction_stream:
        collected_messages.append(extraction.choices[0].delta.content)

    collected_messages = [m for m in collected_messages if m is not None]
    collected_messages = "".join(collected_messages)
    final_output = PartialMeetingInfo.model_validate_json(collected_messages)
    end_time = time.time() - start_time

    print(final_output)
    print(f"Final response received {end_time:.2f} seconds after request")

    output_tokens = num_tokens_from_string(collected_messages, model)
    char_per_sec = output_tokens / end_time
    print(f"{output_tokens} total output tokens")
    print(f"{char_per_sec:.2f} tok/s")

    print("Raw Streaming Benchmark End\n")

def benchmark_partial_streaming(model="gpt-4"):    
    print("Partial Streaming Benchmark Start")

    content = f"Get the information about the meeting and the users {text_block}"

    start_time = time.time()
    extraction_stream = client.chat.completions.create(
        model=model,
        response_model=PartialMeetingInfo,
        messages=[
            {
                "role": "user",
                "content": content
            }
        ],
        stream=True,
    )

    collected_messages = []
    for extraction in extraction_stream:
        collected_messages.append(extraction.__dict__["chunk"])
    end_time = time.time() - start_time

    print(extraction)
    print(f"Final response received {end_time:.2f} seconds after request")

    collected_messages = [m for m in collected_messages if m is not None]
    collected_messages = "".join(collected_messages)

    output_tokens = num_tokens_from_string(collected_messages, model)
    char_per_sec = output_tokens / end_time
    print(f"{output_tokens} total output tokens")
    print(f"{char_per_sec:.2f} tok/s")

    print("Partial Streaming Benchmark End")

benchmark_raw_stream()
"""
Raw Streaming Benchmark Start
users=[PartialUser(name='John Doe', email='johndoe@email.com', twitter='@TechGuru44'), PartialUser(name='Jane Smith', email='janesmith@email.com', twitter='@DigitalDiva88'), PartialUser(name='Alex Johnson', email='alexj@email.com', twitter='@CodeMaster2023')] date='2024-03-15' location='Grand Tech Arena, 4521 Innovation Drive' budget=50000 deadline='2024-02-20'
Final response received 4.74 seconds after request
158 total output tokens
33.35 tok/s
Raw Streaming Benchmark End
"""

benchmark_partial_streaming()
"""
Partial Streaming Benchmark Start
users=[PartialUser(name='John Doe', email='johndoe@email.com', twitter='@TechGuru44'), PartialUser(name='Jane Smith', email='janesmith@email.com', twitter='@DigitalDiva88'), PartialUser(name='Alex Johnson', email='alexj@email.com', twitter='@CodeMaster2023')] date='2024-03-15' location='Grand Tech Arena, 4521 Innovation Drive' budget=50000 deadline='2024-02-20'
Final response received 5.40 seconds after request
156 total output tokens
28.89 tok/s
Partial Streaming Benchmark End
"""
