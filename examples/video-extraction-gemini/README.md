# Video Analysis with Gemini 2.5 Pro

This example demonstrates how to use Google's Gemini 2.5 Pro model with Instructor to analyze videos and extract structured information about tourist destinations.

## Features

- Upload videos to Gemini API using `VideoWithGenaiFile`
- Extract structured recommendations using Pydantic models
- Support for analyzing travel content and tourist destinations
- Type-safe structured outputs

## Requirements

```bash
pip install instructor google-genai pydantic
```

## Setup

1. Get your Google API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Set the environment variable:

```bash
export GOOGLE_API_KEY=your_api_key_here
```

## Usage

Run the script with a path to your video file:

```bash
python run.py path/to/your/video.mp4
```

Example with a travel video:

```bash
python run.py takayama_travel.mp4
```

## How It Works

The example:

1. Uploads your video file to the Gemini API using `VideoWithGenaiFile.from_new_genai_file()`
2. Sends a prompt asking for tourist destination recommendations
3. Uses Instructor to parse the response into structured Pydantic models
4. Returns a list of destinations with names, descriptions, and locations

## Output Structure

The analysis returns:

- **chain_of_thought**: Detailed reasoning about the video content
- **description**: Overall summary of the video
- **destinations**: List of tourist destinations, each with:
  - name: Name of the destination
  - description: What makes it interesting
  - location: Where it's located

## Supported Video Formats

Gemini supports the following video formats:
- MP4
- MPEG
- MOV
- AVI
- FLV
- MPG
- WebM
- WMV
- 3GPP
- QuickTime

## Notes

- Video files are uploaded to Google's servers for processing
- Large videos may take longer to upload and process
- The API automatically waits for the upload to complete before processing

## Related Examples

- [Multimodal Gemini Guide](../../docs/blog/posts/multimodal-gemini.md)
- [Image Analysis with Gemini](../vision/)
- [PDF Processing with Gemini](../../docs/blog/posts/chat-with-your-pdf-with-gemini.md)
