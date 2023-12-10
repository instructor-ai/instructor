from pydantic import BaseModel, Field
from typing import Iterable, List, Literal, Optional
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
    repeats_until: Optional[datetime] = Field(
        default=None,
        description="If the date range repeats, until when does it repeat. This is useful for the case where the date range repeats until a specific date, like a holiday.",
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
                    "explain": "The availability from December 8th to December 24th, 9 am to 5 pm Monday to Saturday, and 10 am to 5 pm on Sunday. This is a one-time event for a specified period.",
                    "start": "2023-12-08T09:00:00",
                    "end": "2023-12-24T17:00:00",
                    "repeats": None,
                    "days_of_week": [
                        "monday",
                        "tuesday",
                        "wednesday",
                        "thursday",
                        "friday",
                        "saturday",
                        "sunday",
                    ],
                    "repeats_until": None,
                },
                {
                    "explain": "On Sundays during the same period, the availability starts at 10 am instead of 9 am.",
                    "start": "2023-12-10T10:00:00",
                    "end": "2023-12-24T17:00:00",
                    "repeats": None,
                    "days_of_week": ["sunday"],
                    "repeats_until": None,
                },
            ]
        }
    {
        "availability": [
            {
                "explain": "The availability is for Fridays, after Thanksgiving, and then Saturdays and Sundays from 9 am until dusk, which we will consider as the end of the standard working day. The phrase 'after Thanksgiving' implies that it is a repeating event every year starting from the day after Thanksgiving.",
                "start": "2023-11-24T09:00:00",
                "end": "2023-11-24T18:00:00",
                "repeats": "weekly",
                "days_of_week": ["friday"],
                "repeats_until": None,
            },
            {
                "explain": "Saturdays and Sundays also have the same availability as Fridays, but starting from the Saturday following Thanksgiving.",
                "start": "2023-11-25T09:00:00",
                "end": "2023-11-25T18:00:00",
                "repeats": "weekly",
                "days_of_week": ["saturday", "sunday"],
                "repeats_until": None,
            },
        ]
    }
