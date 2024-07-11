import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from youtube_transcript_api import YouTubeTranscriptApi


# Define the structure for our chapter data
class Chapter(BaseModel):
    start_ts: float = Field(
        ...,
        description="The start timestamp indicating when the chapter starts in the video.",
    )
    end_ts: float = Field(
        ...,
        description="The end timestamp indicating when the chapter ends in the video.",
    )
    title: str = Field(
        ..., description="A concise and descriptive title for the chapter."
    )
    summary: str = Field(
        ...,
        description="A brief summary of the chapter's content, don't use words like 'the speaker'",
    )


# Initialize the OpenAI client with Instructor
client = instructor.from_openai(OpenAI())


def get_youtube_transcript(video_id: str) -> str:
    """Fetch the transcript of a YouTube video."""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join(
            [f"ts={entry['start']} - {entry['text']}" for entry in transcript]
        )
    except Exception as e:
        print(f"Error fetching transcript: {e}")
        return ""


def extract_chapters(transcript: str):
    """Extract chapters from the transcript using AI."""
    return client.chat.completions.create_iterable(
        model="gpt-4o",  # Using a model with larger context
        response_model=Chapter,
        messages=[
            {
                "role": "system",
                "content": "Analyze the given YouTube transcript and extract chapters. For each chapter, provide a start timestamp, title, summary, and any additional relevant information.",
            },
            {"role": "user", "content": transcript},
        ],
    )


def process_youtube_video(video_id: str):
    # Step 1: Fetch the transcript
    transcript = get_youtube_transcript(video_id)
    if not transcript:
        return "Failed to fetch transcript"

    # Step 2 & 3: Pass to AI model and extract chapters
    chapters = extract_chapters(transcript)

    for chapter in chapters:
        print(chapter.model_dump_json(indent=2))


# Example usage
if __name__ == "__main__":
    # https://www.youtube.com/watch?v=yj-wSRJwrrc
    video_id = "yj-wSRJwrrc"
    process_youtube_video(video_id)
    """ 
    {
  "start_ts": 1.04,
  "end_ts": 14.0,
  "title": "Introduction",
  "summary": "The speaker introduces themselves and the topic of the keynote: using Pydantic to build with language models, focusing on structured prompting and integrating models with existing software systems."
}
{
  "start_ts": 14.719,
  "end_ts": 90.0,
  "title": "Challenges in Using Language Models",
  "summary": "Discussion about the challenges of using language models, such as getting reliable JSON output and integrating with legacy systems. The speaker emphasizes the need for better validation and maintainability."
}
{
  "start_ts": 90.28,
  "end_ts": 188.0,
  "title": "Introducing Pydantic",
  "summary": "Introduction to Pydantic, a Python library for data validation that outputs JSON schema compatible with OpenAI function calling. The benefits include cleaner code and automatic validation."
}
{
  "start_ts": 188.879,
  "end_ts": 383.68,
  "title": "Structured Prompting with Pydantic",
  "summary": "Explanation of structured prompting using Pydantic to define objects and schemas, leading to better code validation, type-checking, and integration with IDE features like auto-complete."
}
{
  "start_ts": 384.44,
  "end_ts": 620.76,
  "title": "Advanced Concepts in Structured Prompting",
  "summary": "Exploration of advanced topics like modularity, Chain of Thought, and reusable components in structured prompting, enhancing code manageability and reliability when using language models."
}
{
  "start_ts": 622.92,
  "end_ts": 718.48,
  "title": "Advanced Applications",
  "summary": "Demonstrating advanced applications such as complex data extraction, planning queries, and building more comprehensive models with structured prompts to achieve reliable outputs."
}
{
  "start_ts": 720.639,
  "end_ts": 1062.919,
  "title": "Future Directions and Opportunities",
  "summary": "Discussion about future opportunities in structured outputs, including multimodal applications and generative UI over various formats like images and audio, making the space exciting for innovation."
}
{
  "start_ts": 1065.19,
  "end_ts": 1073.679,
  "title": "Conclusion and Closing Remarks",
  "summary": "Closing remarks thanking the audience and summarizing key points discussed in the keynote on using structured prompting and Pydantic for better language model integration."
}
"""
