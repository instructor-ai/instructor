from io import StringIO
from typing import Iterable

import pandas as pd
from tomlkit import table
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field

client = instructor.patch(OpenAI(), mode=instructor.function_calls.Mode.MD_JSON)


class Table(BaseModel):
    caption: str = Field(
        description="The caption of the table, should describe the table it describes in detail"
    )
    table_md: str = Field(description="The markdown representation of the table")

    def df(self):
        return (
            pd.read_csv(
                StringIO(self.table_md),  # Get rid of whitespaces
                sep="|",
                index_col=1,
            )
            .dropna(axis=1, how="all")
            .iloc[1:]
            .map(lambda x: x.strip())
        )


def extract_table(url: str) -> Iterable[Table]:
    return client.chat.completions.create(
        model="gpt-4-vision-preview",
        response_model=Iterable[Table],
        max_tokens=1800,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract the table from the image, and describe it. Each table should be tidy, do not try to join tables that should be seperate",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": url},
                    },
                ],
            }
        ],
    )


if __name__ == "__main__":
    url = "https://a.storyblok.com/f/47007/2400x2000/bf383abc3c/231031_uk-ireland-in-three-charts_table_v01_b.png"
    tables = extract_table(url)
    for tbl in tables:
        print(tbl.caption)
        print(tbl.table_md)
        print(tbl.df())
