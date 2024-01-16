import json
import logging
import os
import sys
from typing import Dict, List, Optional

import instructor
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from rich import print as rprint

load_dotenv(find_dotenv())

# Add logger
logging.basicConfig()
logger = logging.getLogger("app")
logger.setLevel("INFO")

class Competitor(BaseModel):
    name: str
    features: Optional[List[str]]


# Define models
class Industry(BaseModel):
    """
    Represents competitors from a specific industry extracted from an image using AI.
    """

    name: str = Field(
        description="the name of the industry for these competitors"
    )
    competitor_list: List[Competitor] = Field(
        description="A dict of competitors where each key is an industry"
    )

class Competition(BaseModel):
    """
    Represents competitors extracted from an image using AI.

    This class serves as a structured representation of 
    competitors and their qualities.
    """

    industry_list: List[Industry] = Field(
        description="A list of industries and their competitors"
    )

# Define clients
client_image = instructor.patch(
    OpenAI(api_key=os.getenv("OPENAI_API_KEY")), mode=instructor.Mode.MD_JSON
)

# Define functions
def read_images(image_urls: List[str]) -> Competition:
    """
    Given a list of image URLs, identify the competitors in the images.
    """

    logger.info(f"Identifying competitors in images... {len(image_urls)} images")

    return client_image.chat.completions.create(
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



def run(images: List[str]) -> Competition:
    """
    Given a list of images, identify the industries and the competitors in the images.
    """

    competitors: Competition = read_images(images)

    return competitors


if __name__ == "__main__":
    # Run logger
    logger.info("Starting app...")

    if len(sys.argv) != 2:
        print("Usage: python app.py <path_to_image_list_file>")
        sys.exit(1)

    image_file = sys.argv[1]
    with open(image_file, "r") as file:
        logger.info(f"Reading images from file: {image_file}")
        try:
            image_list = file.read().splitlines()
            logger.info(f"{len(image_list)} images read from file: {image_file}")
        except Exception as e:
            logger.error(f"Error reading images from file: {image_file}")
            logger.error(e)
            sys.exit(1)

    competitors = run(image_list)

    rprint(f"[green]{len(competitors.industry_list)} industries identified:[/green]")
    for industry in competitors.industry_list:
        rprint(f"[green]{industry.name}[/green]")
        rprint(f"[blue]Features: {industry.competitor_list}[/blue]")

    logger.info("Writing results to file...")

    with open("results.json", "w") as f:
        json.dump(
            {
                "competitors": competitors.model_dump(),
            },
            f,
            indent=4,
        )

""" 
Example output:
{
    "competitors": {
        "industry_list": [
            {
                "name": "Accommodation and Hospitality",
                "competitor_list": [
                    {
                        "name": "craigslist",
                        "features": [
                            "Transactions Offline",
                            "Inexpensive"
                        ]
                    },
                    {
                        "name": "couchsurfing",
                        "features": [
                            "Transactions Offline",
                            "Inexpensive"
                        ]
                    },
                    {
                        "name": "BedandBreakfast.com",
                        "features": [
                            "Transactions Offline",
                            "Inexpensive"
                        ]
                    },
                    {
                        "name": "airbnb",
                        "features": [
                            "Transactions Online",
                            "Inexpensive"
                        ]
                    },
                    {
                        "name": "HOSTELS.com",
                        "features": [
                            "Transactions Online",
                            "Inexpensive"
                        ]
                    },
                    {
                        "name": "VRBO",
                        "features": [
                            "Transactions Offline",
                            "Costly"
                        ]
                    },
                    {
                        "name": "Rentahome",
                        "features": [
                            "Transactions Online",
                            "Costly"
                        ]
                    },
                    {
                        "name": "Orbitz",
                        "features": [
                            "Transactions Online",
                            "Costly"
                        ]
                    },
                    {
                        "name": "Hotels.com",
                        "features": [
                            "Transactions Online",
                            "Costly"
                        ]
                    }
                ]
            },
            {
                "name": "E-commerce Wine Retailers",
                "competitor_list": [
                    {
                        "name": "winesimple",
                        "features": [
                            "Ecommerce Retailers",
                            "True Personalized Selections",
                            "Brand Name Wine",
                            "No Inventory Cost",
                            "Target Mass Market"
                        ]
                    },
                    {
                        "name": "nakedwines.com",
                        "features": [
                            "Ecommerce Retailers",
                            "Target Mass Market"
                        ]
                    },
                    {
                        "name": "Club W",
                        "features": [
                            "Ecommerce Retailers",
                            "Brand Name Wine",
                            "Target Mass Market"
                        ]
                    },
                    {
                        "name": "Tasting Room",
                        "features": [
                            "Ecommerce Retailers",
                            "True Personalized Selections",
                            "Brand Name Wine"
                        ]
                    },
                    {
                        "name": "hellovino",
                        "features": [
                            "Ecommerce Retailers",
                            "True Personalized Selections",
                            "No Inventory Cost",
                            "Target Mass Market"
                        ]
                    }
                ]
            }
        ]
    }
}
"""