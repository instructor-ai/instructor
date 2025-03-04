---
title: Extracting Competitor Data from Slides Using AI
description: Learn how to extract competitor data from presentation slides, leveraging AI for comprehensive information gathering.
---

# Data extraction from slides

In this guide, we demonstrate how to extract data from slides.

!!! tips "Motivation"

   When we want to translate key information from slides into structured data, simply isolating the text and running extraction might not be enough. Sometimes the important data is in the images on the slides, so we should consider including them in our extraction pipeline.
    
## Defining the necessary Data Structures

Let's say we want to extract the competitors from various presentations and categorize them according to their respective industries.

Our data model will have `Industry` which will be a list of `Competitor`'s for a specific industry, and `Competition` which will aggregate the competitors for all the industries.

```python
from pydantic import BaseModel, Field
from typing import Optional, List


class Competitor(BaseModel):
    name: str
    features: Optional[List[str]]


# Define models
class Industry(BaseModel):
    """
    Represents competitors from a specific industry extracted from an image using AI.
    """

    name: str = Field(description="The name of the industry")
    competitor_list: List[Competitor] = Field(
        description="A list of competitors for this industry"
    )


class Competition(BaseModel):
    """
    This class serves as a structured representation of
    competitors and their qualities.
    """

    industry_list: List[Industry] = Field(
        description="A list of industries and their competitors"
    )
```

## Competitors extraction

To extract competitors from slides we will define a function which will read images from urls and extract the relevant information from them.

```python
import instructor
from openai import OpenAI

# Apply the patch to the OpenAI client
# enables response_model keyword
client = instructor.from_openai(OpenAI())
# <%hide%>
from pydantic import BaseModel, Field
from typing import Optional, List


class Competitor(BaseModel):
    name: str
    features: Optional[List[str]]


# Define models
class Industry(BaseModel):
    """
    Represents competitors from a specific industry extracted from an image using AI.
    """

    name: str = Field(description="The name of the industry")
    competitor_list: List[Competitor] = Field(
        description="A list of competitors for this industry"
    )


class Competition(BaseModel):
    """
    This class serves as a structured representation of
    competitors and their qualities.
    """

    industry_list: List[Industry] = Field(
        description="A list of industries and their competitors"
    )


# <%hide%>


# Define functions
def read_images(image_urls: List[str]) -> Competition:
    """
    Given a list of image URLs, identify the competitors in the images.
    """
    return client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=Competition,
        max_tokens=2048,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Identify competitors and generate key features for each competitor.",
                    },
                    *[
                        {"type": "image_url", "image_url": {"url": url}}
                        for url in image_urls
                    ],
                ],
            }
        ],
    )
```

## Execution

Finally, we will run the previous function with a few sample slides to see the data extractor in action.

As we can see, our model extracted the relevant information for each competitor regardless of how this information was formatted in the original presentations.

```python
# <%hide%>
import instructor
from openai import OpenAI

# Apply the patch to the OpenAI client
# enables response_model keyword
client = instructor.from_openai(OpenAI())
from pydantic import BaseModel, Field
from typing import Optional, List


class Competitor(BaseModel):
    name: str
    features: Optional[List[str]]


# Define models
class Industry(BaseModel):
    """
    Represents competitors from a specific industry extracted from an image using AI.
    """

    name: str = Field(description="The name of the industry")
    competitor_list: List[Competitor] = Field(
        description="A list of competitors for this industry"
    )


class Competition(BaseModel):
    """
    This class serves as a structured representation of
    competitors and their qualities.
    """

    industry_list: List[Industry] = Field(
        description="A list of industries and their competitors"
    )


# Define functions
def read_images(image_urls: List[str]) -> Competition:
    """
    Given a list of image URLs, identify the competitors in the images.
    """
    return client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=Competition,
        max_tokens=2048,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Identify competitors and generate key features for each competitor.",
                    },
                    *[
                        {"type": "image_url", "image_url": {"url": url}}
                        for url in image_urls
                    ],
                ],
            }
        ],
    )


# <%hide%>
url = [
    'https://miro.medium.com/v2/resize:fit:1276/0*h1Rsv-fZWzQUyOkt',
]
model = read_images(url)
print(model.model_dump_json(indent=2))
"""
{
  "industry_list": [
    {
      "name": "Accommodation Services",
      "competitor_list": [
        {
          "name": "CouchSurfing",
          "features": [
            "Free accommodation",
            "Cultural exchange",
            "Community-driven",
            "User profiles and reviews"
          ]
        },
        {
          "name": "Craigslist",
          "features": [
            "Local listings",
            "Variety of accommodation types",
            "Direct communication with hosts",
            "No booking fees"
          ]
        },
        {
          "name": "BedandBreakfast.com",
          "features": [
            "Specialized in B&Bs",
            "User reviews",
            "Booking options",
            "Local experiences"
          ]
        },
        {
          "name": "AirBed & Breakfast (Airbnb)",
          "features": [
            "Wide range of accommodations",
            "User reviews",
            "Instant booking",
            "Host profiles"
          ]
        },
        {
          "name": "Hostels.com",
          "features": [
            "Budget-friendly hostels",
            "User reviews",
            "Booking options",
            "Global reach"
          ]
        },
        {
          "name": "RentDigs.com",
          "features": [
            "Rental listings",
            "User-friendly interface",
            "Local listings",
            "Direct communication with landlords"
          ]
        },
        {
          "name": "VRBO",
          "features": [
            "Vacation rentals",
            "Family-friendly options",
            "User reviews",
            "Booking protection"
          ]
        },
        {
          "name": "Hotels.com",
          "features": [
            "Wide range of hotels",
            "Rewards program",
            "User reviews",
            "Price match guarantee"
          ]
        }
      ]
    }
  ]
}
"""
```
