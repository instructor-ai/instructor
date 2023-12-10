from pydantic import BaseModel, Field
from typing import Iterable, List, Literal
from datetime import datetime, timedelta

from openai import OpenAI
import instructor

client = instructor.patch(OpenAI())


class DateRange(BaseModel):
    explain: str = Field(
        ...,
        description="Explain the  date range in the context of the text before generating the date range and the repeat pattern.",
    )
    start: datetime
    end: datetime
    repeats: Literal["daily", "weekly", "monthly", None] = Field(
        default=None,
        description="If the date range repeats, and how often, this way we can generalize the date range to the future., if its special, then we can assume it is a one time event.",
    )
    days_of_week: List[
        Literal[
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
            None,
        ]
    ] = Field(
        ...,
        description="If the date range repeats, which days of the week does it repeat on.",
    )


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
                    "explain": "Availability from December 8th to December 24th, opening at 9am and closing at 5pm from Monday to Saturday.",
                    "start": "2023-12-08T09:00:00",
                    "end": "2023-12-08T17:00:00",
                    "repeats": "daily",
                },
                {
                    "explain": "On Sundays during the same period, the opening hours are 10am to 5pm.",
                    "start": "2023-12-10T10:00:00",
                    "end": "2023-12-10T17:00:00",
                    "repeats": "weekly",
                },
                {
                    "explain": "Availability from December 8th to December 24th, opening at 9am and closing at 5pm from Monday to Saturday, excluding Sundays.",
                    "start": "2023-12-11T09:00:00",
                    "end": "2023-12-11T17:00:00",
                    "repeats": "weekly",
                },
                {
                    "explain": "Repeated availability every Monday to Saturday excluding the first Sunday.",
                    "start": "2023-12-12T09:00:00",
                    "end": "2023-12-12T17:00:00",
                    "repeats": "daily",
                },
            ]
        }
{
    "availability": [
        {
            "explain": "Open the Friday after Thanksgiving, which would vary each year, so we will need the date for the specific year to provide the exact range.",
            "start": "2023-11-24T09:00:00",
            "end": "2023-11-24T17:00:00",
            "repeats": null,
        },
        {
            "explain": "Open on Saturdays and Sundays from 9am until dusk. Assuming dusk to be at 5pm, which may vary throughout the year.",
            "start": "2023-12-16T09:00:00",
            "end": "2023-12-16T17:00:00",
            "repeats": "weekly",
        },
        {
            "explain": "Continued availability every Saturday and Sunday starting from December 16th.",
            "start": "2023-12-17T09:00:00",
            "end": "2023-12-17T17:00:00",
            "repeats": "weekly",
        },
    ]
}
