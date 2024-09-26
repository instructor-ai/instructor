import instructor
import google.generativeai as genai
from pydantic import BaseModel
import os


class Description(BaseModel):
    relevant_speakers: list[str]
    summary: str


curr_file = os.path.dirname(__file__)
file_path = os.path.join(curr_file, "./test_files/sample.mp3")


def test_audio_compatability_list():
    client = instructor.from_gemini(
        genai.GenerativeModel("gemini-1.5-flash-latest"),
        mode=instructor.Mode.GEMINI_JSON,
    )

    files = [file for file in genai.list_files()]
    file_names = [file.display_name for file in files]

    if "sample.mp3" not in file_names:
        file = genai.upload_file(file_path)
    else:
        print("File already uploaded, extracting file obj now")
        file = [file for file in files if file.display_name == "sample.mp3"][0]

    content = ["Please transcribe this recording:", file]

    result = client.chat.completions.create(
        response_model=Description,
        messages=[
            {"role": "user", "content": content},
        ],
    )

    assert isinstance(
        result, Description
    ), "Result should be an instance of Description"


def test_audio_compatability_multiple_messages():
    client = instructor.from_gemini(
        genai.GenerativeModel("gemini-1.5-flash-latest"),
        mode=instructor.Mode.GEMINI_JSON,
    )

    files = [file for file in genai.list_files()]
    file_names = [file.display_name for file in files]

    if "sample.mp3" not in file_names:
        file = genai.upload_file(file_path)
    else:
        print("File already uploaded, extracting file obj now")
        file = [file for file in files if file.display_name == "sample.mp3"][0]

    result = client.chat.completions.create(
        response_model=Description,
        messages=[
            {"role": "user", "content": "Please transcribe this recording:"},
            {"role": "user", "content": file},
        ],
    )

    assert isinstance(
        result, Description
    ), "Result should be an instance of Description"
