# Data extraction from slides

In this guide, we demonstrate how to extract data from slides.

!!! tips "Motivation"

   When we want to translate key information from slides into structured data, simply isolating the text and running extraction might not be enough. Sometimes the important data is in the images on the slides, so we should consider including them in our extraction pipeline.
    
## Defining the necessary Data Structures

Let's say we want to extract the competitors from various presentations and categorize them according to their respective industries.

Our data model will have `Industry` which will be a list of `Competitor`'s for a specific industry, and `Competition` which will aggregate the competitors for all the industries.

```python
from openai import OpenAI
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

    name: str = Field(
        description="The name of the industry"
    )
    competitor_list: List[Competitor] = Field(
        description="A list of competitors for this industry"
    )

class Competition(BaseModel):
    """
    This class serves as a structured representation of 
    competitors and their qualities.
    """

    industry_list: List[IndustryCompetition] = Field(
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
client = instructor.from_openai(
    OpenAI(), mode=instructor.Mode.MD_JSON
)

# Define functions
def read_images(image_urls: List[str]) -> Competition:
    """
    Given a list of image URLs, identify the competitors in the images.
    """
    return client.chat.completions.create(
        model="gpt-4-vision-preview",
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
url = [
    'https://miro.medium.com/v2/resize:fit:1276/0*h1Rsv-fZWzQUyOkt', 
    'https://earlygame.vc/wp-content/uploads/2020/06/startup-pitch-deck-5.jpg'
    ]
model = read_images(url)
print(model.model_json_dump(indent=2))
```
    industry_list=[

    Industry(name='Accommodation and Hospitality', competitor_list=[Competitor(name='CouchSurfing', features=['Affordable', 'Online Transaction']), Competitor(name='Craigslist', features=['Affordable', 'Offline Transaction']), Competitor(name='BedandBreakfast.com', features=['Affordable', 'Offline Transaction']), Competitor(name='AirBed&Breakfast', features=['Affordable', 'Online Transaction']), Competitor(name='Hostels.com', features=['Affordable', 'Online Transaction']), Competitor(name='VRBO', features=['Expensive', 'Offline Transaction']), Competitor(name='Rentahome', features=['Expensive', 'Online Transaction']), Competitor(name='Orbitz', features=['Expensive', 'Online Transaction']), Competitor(name='Hotels.com', features=['Expensive', 'Online Transaction'])]), 
    
    Industry(name='Wine E-commerce', competitor_list=[Competitor(name='WineSimple', features=['Ecommerce Retailers', 'True Personalized Selections', 'Brand Name Wine', 'No Inventory Cost', 'Target Mass Market']), Competitor(name='NakedWines', features=['Ecommerce Retailers', 'Target Mass Market']), Competitor(name='Club W', features=['Ecommerce Retailers', 'Brand Name Wine', 'Target Mass Market']), Competitor(name='Tasting Room', features=['Ecommerce Retailers', 'True Personalized Selections', 'Brand Name Wine']), Competitor(name='Drync', features=['Ecommerce Retailers', 'True Personalized Selections', 'No Inventory Cost']), Competitor(name='Hello Vino', features=['Ecommerce Retailers', 'Brand Name Wine', 'Target Mass Market'])])

    ]
```
```
