---
draft: False
date: 2024-07-11
slug: youtube-transcripts
comments: true
authors:
  - jxnl
---

# YouTube Transcript Analysis with Instructor and Pydantic

If you want to build applications that process transcripts, youtube video is a great place to start!

In this post, we'll walk through a complete example of how to extract structured chapter information from YouTube video transcripts using OpenAI's language models, the Instructor library, and Pydantic. Then, we'll explore some alternative ideas for different applications.

## Part 1: Generating Chapters - A Complete Example

Let's start with a step-by-step guide to generating chapters from a YouTube video transcript.

### Step 1: Install Required Libraries

First, make sure you have all the necessary libraries installed:

```bash
pip install openai instructor pydantic youtube_transcript_api
```

### Step 2: Import Libraries and Set Up OpenAI Client

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from youtube_transcript_api import YouTubeTranscriptApi
import os

# Set up OpenAI client
client = instructor.from_openai(OpenAI())

# Ensure you've set your OpenAI API key in your environment variables
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"
```

### Step 3: Define the Chapter Model

We'll use Pydantic to define our chapter structure:

```python
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
```

### Step 4: Create Functions to Fetch Transcript and Extract Chapters

```python
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
        model="gpt-4",  # You can experiment with different models
        response_model=Chapter,
        messages=[
            {
                "role": "system",
                "content": "Analyze the given YouTube transcript and extract chapters. For each chapter, provide a start timestamp, end timestamp, title, and summary.",
            },
            {"role": "user", "content": transcript},
        ],
    )
```

### Step 5: Create the Main Processing Function

```python
def process_youtube_video(video_id: str):
    """Process a YouTube video to extract chapters."""
    transcript = get_youtube_transcript(video_id)
    if not transcript:
        return "Failed to fetch transcript"

    chapters = extract_chapters(transcript)

    for chapter in chapters:
        print(chapter.model_dump_json(indent=2))
```

### Step 6: Run the Script

```python
if __name__ == "__main__":
    video_id = "yj-wSRJwrrc"  # Replace with your YouTube video ID
    process_youtube_video(video_id)
```

![Example Output](./img/youtube-chapters.png)

### Complete Script

Here's the complete script all together:

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from youtube_transcript_api import YouTubeTranscriptApi
import os

# Set up OpenAI client
client = instructor.from_openai(OpenAI())

# Ensure you've set your OpenAI API key in your environment variables
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"

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
        model="gpt-4",  # You can experiment with different models
        response_model=Chapter,
        messages=[
            {
                "role": "system",
                "content": "Analyze the given YouTube transcript and extract chapters. For each chapter, provide a start timestamp, end timestamp, title, and summary.",
            },
            {"role": "user", "content": transcript},
        ],
    )

def process_youtube_video(video_id: str):
    """Process a YouTube video to extract chapters."""
    transcript = get_youtube_transcript(video_id)
    if not transcript:
        return "Failed to fetch transcript"

    chapters = extract_chapters(transcript)

    for chapter in chapters:
        print(chapter.model_dump_json(indent=2))

if __name__ == "__main__":
    video_id = "yj-wSRJwrrc"  # Replace with your YouTube video ID
    process_youtube_video(video_id)
```

This script will fetch the transcript of the specified YouTube video, use AI to extract chapters from the transcript, and print the structured chapter information.

## Alternative Ideas with Different Models

Now that we've seen a complete example of chapter extraction, let's explore some alternative ideas using different Pydantic models. These models can be used to adapt our YouTube transcript analysis for various applications.

### 1. Study Notes Generator

```python
from pydantic import BaseModel, Field
from typing import List

class Concept(BaseModel):
    term: str = Field(..., description="A key term or concept mentioned in the video")
    definition: str = Field(..., description="A brief definition or explanation of the term")

class StudyNote(BaseModel):
    timestamp: float = Field(..., description="The timestamp where this note starts in the video")
    topic: str = Field(..., description="The main topic being discussed at this point")
    key_points: List[str] = Field(..., description="A list of key points discussed")
    concepts: List[Concept] = Field(..., description="Important concepts mentioned in this section")
```

This model structures the video content into clear topics, key points, and important concepts, making it ideal for revision and study purposes.

### 2. Content Summarization

```python
from pydantic import BaseModel, Field
from typing import List

class ContentSummary(BaseModel):
    title: str = Field(..., description="The title of the video")
    duration: float = Field(..., description="The total duration of the video in seconds")
    main_topics: List[str] = Field(..., description="A list of main topics covered in the video")
    key_takeaways: List[str] = Field(..., description="The most important points from the entire video")
    target_audience: str = Field(..., description="The intended audience for this content")
```

This model provides a high-level overview of the entire video, perfect for quick content analysis or deciding whether a video is worth watching in full.

### 3. Quiz Generator

```python
from pydantic import BaseModel, Field
from typing import List

class QuizQuestion(BaseModel):
    question: str = Field(..., description="The quiz question")
    options: List[str] = Field(..., min_items=2, max_items=4, description="Possible answers to the question")
    correct_answer: int = Field(..., ge=0, lt=4, description="The index of the correct answer in the options list")
    explanation: str = Field(..., description="An explanation of why the correct answer is correct")

class VideoQuiz(BaseModel):
    title: str = Field(..., description="The title of the quiz, based on the video content")
    questions: List[QuizQuestion] = Field(..., min_items=5, max_items=20, description="A list of quiz questions based on the video content")
```

This model transforms video content into an interactive quiz, perfect for testing comprehension or creating engaging content for social media.

To use these alternative models, you would replace the `Chapter` model in our original code with one of these alternatives and adjust the system prompt in the `extract_chapters` function accordingly.

## Conclusion

The power of this approach lies in its flexibility. By defining different Pydantic models, we can repurpose our YouTube transcript analysis for a wide variety of applications. Whether you're creating study materials, improving accessibility, optimizing for SEO, or generating quizzes, the combination of Instructor, Pydantic, and AI language models provides a robust framework for extracting structured information from video content.

Remember to experiment with different models and fine-tune your prompts to get the best results for your specific use case. The possibilities are endless, and with Pydantic, implementing new ideas is just a model definition away!

Leave a comment below if you have any questions or feedback.