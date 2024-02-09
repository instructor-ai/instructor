from openai import OpenAI
from io import StringIO
from typing import Annotated, Any, Iterable
from pydantic import (
    BaseModel,
    BeforeValidator,
    PlainSerializer,
    InstanceOf,
    WithJsonSchema,
)
import pandas as pd
import instructor


client = instructor.patch(OpenAI(), mode=instructor.function_calls.Mode.MD_JSON)


def md_to_df(data: Any) -> Any:
    if isinstance(data, str):
        return (
            pd.read_csv(
                StringIO(data),  # Get rid of whitespaces
                sep="|",
                index_col=1,
            )
            .dropna(axis=1, how="all")
            .iloc[1:]
            .map(lambda x: x.strip())
        )
    return data


MarkdownDataFrame = Annotated[
    InstanceOf[pd.DataFrame],
    BeforeValidator(md_to_df),
    PlainSerializer(lambda x: x.to_markdown()),
    WithJsonSchema(
        {
            "type": "string",
            "description": """
                The markdown representation of the table, 
                each one should be tidy, do not try to join tables
                that should be seperate""",
        }
    ),
]


class Table(BaseModel):
    caption: str
    dataframe: MarkdownDataFrame


tables = client.chat.completions.create(
    model="gpt-4-vision-preview",
    max_tokens=1000,
    response_model=Iterable[Table],
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe this data accurately as a table in markdown format.",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        # "url": "https://a.storyblok.com/f/47007/2400x1260/f816b031cb/uk-ireland-in-three-charts_chart_a.png/m/2880x0",
                        # "url": "https://a.storyblok.com/f/47007/2400x2000/bf383abc3c/231031_uk-ireland-in-three-charts_table_v01_b.png/m/2880x0",
                        # "url": "https://a.storyblok.com/f/47007/4800x2766/1688e25601/230629_attoptinratesmidyear_blog_chart02_v01.png/m/2880x0"
                        "url": "https://a.storyblok.com/f/47007/2400x1260/934d294894/uk-ireland-in-three-charts_chart_b.png/m/2880x0"
                    },
                },
                {
                    "type": "text",
                    "text": """
                        First take a moment to reason about the best set of headers for the tables. 
                        Write a good h1 for the image above. Then follow up with a short description of the what the data is about.
                        Then for each table you identified, write a h2 tag that is a descriptive title of the table. 
                        Then follow up with a short description of the what the data is about. 
                        Lastly, produce the markdown table for each table you identified.
                    """,
                },
            ],
        }
    ],
)

for table in tables:
    print(table.caption)
    print(table.dataframe)
    print()
    """
    D1 App Retention Rates July 2023 (Ireland & U.K.)
                    Ireland   UK  
    Category                       
    Education             14%   12%
    Entertainment         13%   11%
    Games                 26%   25%
    Social                27%   18%
    Utilities             11%    9%
    """
