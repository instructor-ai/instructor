---
authors:
  - ivanleomk
categories:
  - Gemini
  - Multimodal
comments: true
date: 2024-10-23
description: Learn how to use Google's Gemini model for multimodal structured extraction of YouTube videos, extracting structured recommendations for tourist destinations.
draft: false
tags:
  - Gemini
  - Multimodal AI
  - Travel Recommendations
  - Pydantic
  - Python
---

# Structured Outputs with Multimodal Gemini

In this post, we'll explore how to use Google's Gemini model with Instructor to analyze [travel videos](https://www.youtube.com/watch?v=_R8yhW_H9NQ) and extract structured recommendations. This powerful combination allows us to process multimodal inputs (video) and generate structured outputs using Pydantic models. This post was done in collaboration with [Kino.ai](https://kino.ai), a company that uses instructor to do structured extraction from multimodal inputs to improve search for film makers.

## Setting Up the Environment

First, install the required dependencies:

```bash
pip install "instructor[google-genai]" pydantic
```

<!-- more -->

Make sure you have your Google API key set as an environment variable:

```bash
export GOOGLE_API_KEY=your_api_key_here
```

## Defining Our Data Models

We'll use Pydantic to define our data models for tourist destinations and recommendations:

```python
class TouristDestination(BaseModel):
    name: str
    description: str
    location: str


class Recommendations(BaseModel):
    chain_of_thought: str
    description: str
    destinations: list[TouristDestination]
```

## Initializing the Gemini Client

Next, set up the Gemini client using Instructor's unified provider interface:

```python
import instructor

client = instructor.from_provider(
    "google/gemini-2.0-flash-exp",
    async_client=False,
)
```

## Uploading and Processing the Video

To analyze a video, first upload it using `VideoWithGenaiFile`:

```python
# Upload the video and wait for processing
video = instructor.VideoWithGenaiFile.from_new_genai_file("./takayama.mp4")
```

Then, process the video and extract recommendations:

```python
resp = client.messages.create(
    messages=[
        {
            "role": "user",
            "content": [
                "What places do they recommend in this video?",
                video,
            ],
        }
    ],
    response_model=Recommendations,
)

print(resp)
```

??? note "Expand to see Raw Results"

    ```python
    Recomendations(
        chain_of_thought='The video recommends visiting Takayama city, in the Hida Region, Gifu Prefecture. The
    video suggests visiting the Miyagawa Morning Market, to try the Sarubobo good luck charms, and to enjoy the
    cookie cup espresso, made by Koma Coffee. Then, the video suggests visiting a traditional Japanese Cafe,
    called Kissako Katsure, and try their matcha and sweets. Afterwards, the video suggests to visit the Sanmachi
    Historic District, where you can find local crafts and delicious foods. The video recommends trying Hida Wagyu
    beef, at the Kin no Kotte Ushi shop, or to have a sit-down meal at the Kitchen Hida. Finally, the video
    recommends visiting Shirakawa-go, a World Heritage Site in Gifu Prefecture.',
        description='This video recommends a number of places to visit in Takayama city, in the Hida Region, Gifu
    Prefecture. It shows some of the local street food and highlights some of the unique shops and restaurants in
    the area.',
        destinations=[
            TouristDestination(
                name='Takayama',
                description='Takayama is a city at the base of the Japan Alps, located in the Hida Region of
    Gifu.',
                location='Hida Region, Gifu Prefecture'
            ),
            TouristDestination(
                name='Miyagawa Morning Market',
                description="The Miyagawa Morning Market, or the Miyagawa Asai-chi in Japanese, is a market that
    has existed officially since the Edo Period, more than 100 years ago. It's open every single day, rain or
    shine, from 7am to noon.",
                location='Hida Takayama'
            ),
            TouristDestination(
                name='Nakaya - Handmade Hida Sarubobo',
                description='The Nakaya shop sells handcrafted Sarubobo good luck charms.',
                location='Hida Takayama'
            ),
            TouristDestination(
                name='Koma Coffee',
                description="Koma Coffee is a shop that has been in business for about 50 or 60 years, and they
    serve coffee in a cookie cup. They've been serving coffee for about 10 years.",
                location='Hida Takayama'
            ),
            TouristDestination(
                name='Kissako Katsure',
                description='Kissako Katsure is a traditional Japanese style cafe, called Kissako, and the name
    means would you like to have some tea. They have a variety of teas and sweets.',
                location='Hida Takayama'
            ),
            TouristDestination(
                name='Sanmachi Historic District',
                description='Sanmachi Dori is a Historic Merchant District in Takayama, all of the buildings here
    have been preserved to look as they did in the Edo Period.',
                location='Hida Takayama'
            ),
            TouristDestination(
                name='Suwa Orchard',
                description='The Suwa Orchard has been in business for more than 50 years.',
                location='Hida Takayama'
            ),
            TouristDestination(
                name='Kitchen HIDA',
                description='Kitchen HIDA is a restaurant with a 50 year history, known for their Hida Beef dishes
    and for using a lot of local ingredients.',
                location='Hida Takayama'
            ),
            TouristDestination(
                name='Kin no Kotte Ushi',
                description='Kin no Kotte Ushi is a shop known for selling Beef Sushi, especially Hida Wagyu Beef
    Sushi. Their sushi is medium rare.',
                location='Hida Takayama'
            ),
            TouristDestination(
                name='Shirakawa-go',
                description='Shirakawa-go is a World Heritage Site in Gifu Prefecture.',
                location='Gifu Prefecture'
            )
        ]
    )
    ```

The Gemini model analyzes the video and provides structured recommendations. Here's a summary of the extracted information:

1. **Takayama City**: The main destination, located in the Hida Region of Gifu Prefecture.
2. **Miyagawa Morning Market**: A historic market open daily from 7am to noon.
3. **Nakaya Shop**: Sells handcrafted Sarubobo good luck charms.
4. **Koma Coffee**: A 50-60 year old shop famous for serving coffee in cookie cups.
5. **Kissako Katsure**: A traditional Japanese cafe offering various teas and sweets.
6. **Sanmachi Historic District**: A preserved merchant district from the Edo Period.
7. **Suwa Orchard**: A 50+ year old orchard business.
8. **Kitchen HIDA**: A restaurant with a 50-year history, known for Hida Beef dishes.
9. **Kin no Kotte Ushi**: A shop specializing in Hida Wagyu Beef Sushi.
10. **Shirakawa-go**: A World Heritage Site in Gifu Prefecture.

## Limitations, Challenges, and Future Directions

While the current approach demonstrates the power of multimodal AI for video analysis, there are several limitations and challenges to consider:

1. **Lack of Temporal Information**: Our current method extracts overall recommendations but doesn't provide timestamps for specific mentions. This limits the ability to link recommendations to exact moments in the video.

2. **Speaker Diarization**: The model doesn't distinguish between different speakers in the video. Implementing speaker diarization could provide valuable context about who is making specific recommendations.

3. **Content Density**: Longer or more complex videos might overwhelm the model, potentially leading to missed information or less accurate extractions.

### Future Explorations

To address these limitations and expand the capabilities of our video analysis system, here are some promising areas to explore:

1. **Timestamp Extraction**: Enhance the model to provide timestamps for each recommendation or point of interest mentioned in the video. This could be achieved by:

   ```python
   class TimestampedRecommendation(BaseModel):
       timestamp: str
       timestamp_format: Literal["HH:MM", "HH:MM:SS"]  # Helps with parsing
       recommendation: str


   class EnhancedRecommendations(BaseModel):
       destinations: list[TouristDestination]
       timestamped_mentions: list[TimestampedRecommendation]
   ```

2. **Speaker Diarization**: Implement speaker recognition to attribute recommendations to specific individuals. This could be particularly useful for videos featuring multiple hosts or interviewees.

3. **Segment-based Analysis**: Process longer videos in segments to maintain accuracy and capture all relevant information. This approach could involve:

   - Splitting the video into smaller chunks
   - Analyzing each chunk separately
   - Aggregating and deduplicating results

4. **Multi-language Support**: Extend the model's capabilities to accurately analyze videos in various languages and capture culturally specific recommendations.

5. **Visual Element Analysis**: Enhance the model to recognize and describe visual elements like landmarks, food dishes, or activities shown in the video, even if not explicitly mentioned in the audio.

6. **Sentiment Analysis**: Incorporate sentiment analysis to gauge the speaker's enthusiasm or reservations about specific recommendations.

By addressing these challenges and exploring these new directions, we can create a more comprehensive and nuanced video analysis system, opening up even more possibilities for applications in travel, education, and beyond.

## Complete Working Example

Here's a complete, runnable example that you can use with your own videos:

```python
import instructor
from pydantic import BaseModel


class TouristDestination(BaseModel):
    name: str
    description: str
    location: str


class Recommendations(BaseModel):
    chain_of_thought: str
    description: str
    destinations: list[TouristDestination]


def analyze_video(video_path: str):
    # Initialize the client
    client = instructor.from_provider(
        "google/gemini-2.0-flash-exp",
        async_client=False,
    )
    
    # Upload the video
    video = instructor.VideoWithGenaiFile.from_new_genai_file(video_path)
    
    # Extract structured data
    recommendations = client.messages.create(
        messages=[
            {
                "role": "user",
                "content": [
                    "What tourist destinations do they recommend in this video?",
                    video,
                ],
            }
        ],
        response_model=Recommendations,
    )
    
    return recommendations


# Usage
results = analyze_video("./travel_video.mp4")
for dest in results.destinations:
    print(f"{dest.name} - {dest.location}")
    print(f"  {dest.description}\n")
```

For a more detailed example with proper error handling and formatting, check out our [video extraction example](https://github.com/instructor-ai/instructor/tree/main/examples/video-extraction-gemini).

## Related Documentation
- [Multimodal Concepts](../../concepts/multimodal.md) - Working with images, video, and audio
- [Google Integration](../../integrations/google.md) - Complete Gemini setup guide
- [Video Extraction Example](https://github.com/instructor-ai/instructor/tree/main/examples/video-extraction-gemini) - Complete working example with video

## See Also
- [OpenAI Multimodal](openai-multimodal.md) - Compare multimodal approaches
- [Anthropic Structured Output](structured-output-anthropic.md) - Alternative provider
- [Chat with PDFs using Gemini](chat-with-your-pdf-with-gemini.md) - Practical PDF processing
