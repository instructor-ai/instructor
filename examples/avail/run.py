from pydantic import BaseModel, Field
from typing import Iterable, List
from datetime import datetime, timedelta

from openai import OpenAI
import instructor

client = instructor.patch(OpenAI())


class DateRange(BaseModel):
    explination: str = Field(
        ..., description="Explination of the date range in the context of the text"
    )
    start: datetime
    end: datetime


class AvailabilityResponse(BaseModel):
    availability: List[DateRange]


text = """
#1
12/8-12/24
9am - 5pm Monday - Saturday
10am - 5pm Sunday
#2
We are open Friday, after Thanksgiving, and then Saturdays and Sundays 9 a.m. till dusk.``
"""


def prepare_dates(n=7) -> str:
    # Current date and time
    now = datetime.now()

    acc = ""
    # Loop for the next 7 days
    for i in range(n):
        # Calculate the date for each day
        day = now + timedelta(days=i)
        # Print the day of the week, date, and time
        acc += "\n" + day.strftime("%A, %Y-%m-%d %H:%M:%S")

    return acc.strip()


def parse_availability(text: str) -> Iterable[AvailabilityResponse]:
    return client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {
                "role": "system",
                "content": "You are a state of the art date range parse designed to correctly extract availabilities.",
            },
            {
                "role": "user",
                "content": text,
            },
            {
                "role": "user",
                "content": f"To help you understand the dates, here are the next 7 days: {prepare_dates()}",
            },
        ],
        response_model=Iterable[AvailabilityResponse],
    )


if __name__ == "__main__":
    schedules = parse_availability(text)
    for schedule in schedules:
        print(schedule.model_dump_json(indent=2))
    {
        "availability": [
            {
                "explination": "General availability from December 8th to December 24th, 9am - 5pm Monday to Saturday",
                "start": "2023-12-11T09:00:00",
                "end": "2023-12-16T17:00:00",
            },
            {
                "explination": "General availability for the Sundays within the range of December 8th to December 24th, 10am - 5pm.",
                "start": "2023-12-10T10:00:00",
                "end": "2023-12-10T17:00:00",
            },
            {
                "explination": "Continuation of general availability from December 8th to December 24th, 9am - 5pm Monday to Saturday for the following week.",
                "start": "2023-12-18T09:00:00",
                "end": "2023-12-23T17:00:00",
            },
            {
                "explination": "Continuation of general availability for the last Sunday within the range of December 8th to December 24th, 10am - 5pm.",
                "start": "2023-12-17T10:00:00",
                "end": "2023-12-17T17:00:00",
            },
        ]
    }
{
    "availability": [
        {
            "explination": "Special availability for the Friday after Thanksgiving, assumed to be November 24th, 2023 based on the given next days.",
            "start": "2023-11-24T09:00:00",
            "end": "2023-11-24T17:00:00",
        },
        {
            "explination": "Saturdays availability from after Thanksgiving up to the foreseeable Saturdays, 9am till dusk (assumed to be 5pm based on the general availability previously mentioned).",
            "start": "2023-11-25T09:00:00",
            "end": "2023-11-25T17:00:00",
        },
        {
            "explination": "Sundays availability from after Thanksgiving up to the foreseeable Sundays, 9am till dusk (assumed to be 5pm).",
            "start": "2023-12-10T09:00:00",
            "end": "2023-12-10T17:00:00",
        },
        {
            "explination": "Continuation of Saturdays availability, 9am till dusk.",
            "start": "2023-12-16T09:00:00",
            "end": "2023-12-16T17:00:00",
        },
    ]
}
