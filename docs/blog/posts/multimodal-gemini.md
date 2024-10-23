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

First, let's set up our environment with the necessary libraries:

```python
from pydantic import BaseModel
import instructor
import google.generativeai as genai
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

Next, we'll set up our Gemini client using Instructor:

```python
client = instructor.from_gemini(
    client=genai.GenerativeModel(
        model_name="models/gemini-1.5-flash-latest",
    ),
)
```

## Uploading and Processing the Video

To analyze a video, we first need to upload it:

```python
file = genai.upload_file("./takayama.mp4")
```

Then, we can process the video and extract recommendations:

```python
resp = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": ["What places do they recommend in this video?", file],
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

## Conclusion

This example demonstrates the power of combining multimodal AI with structured output parsing. By using Gemini with Instructor, we can extract rich, structured information from video content, opening up new possibilities for travel recommendation systems, content analysis, and more.

The ability to process video inputs and generate structured data outputs can be applied to various domains beyond travel, such as education, entertainment, and market research. As multimodal AI continues to evolve, we can expect even more sophisticated applications that bridge the gap between visual content and structured data.