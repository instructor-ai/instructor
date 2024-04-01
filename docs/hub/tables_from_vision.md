# Extracting Tables from Images with OpenAI's GPT-4 Vision Model

First, we define a custom type, `MarkdownDataFrame`, to handle pandas DataFrames formatted in markdown. This type uses Python's `Annotated` and `InstanceOf` types, along with decorators `BeforeValidator` and `PlainSerializer`, to process and serialize the data.

## Defining the Table Class

The `Table` class is essential for organizing the extracted data. It includes a caption and a dataframe, processed as a markdown table. Since most of the complexity is handled by the `MarkdownDataFrame` type, the `Table` class is straightforward!

This requires additional dependencies `pip install pandas tabulate`.

```python
from openai import OpenAI
from io import StringIO
from typing import Annotated, Any, List
from pydantic import (
    BaseModel,
    BeforeValidator,
    PlainSerializer,
    InstanceOf,
    WithJsonSchema,
)
import instructor
import pandas as pd


client = instructor.from_openai(OpenAI(), mode=instructor.Mode.MD_JSON)


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


class MultipleTables(BaseModel):
    tables: List[Table]


example = MultipleTables(
    tables=[
        Table(
            caption="This is a caption",
            dataframe=pd.DataFrame(
                {
                    "Chart A": [10, 40],
                    "Chart B": [20, 50],
                    "Chart C": [30, 60],
                }
            ),
        )
    ]
)


def extract(url: str) -> MultipleTables:
    tables = client.chat.completions.create(
        model="gpt-4-vision-preview",
        max_tokens=4000,
        response_model=MultipleTables,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Describe this data accurately as a table in markdown format. {example.model_dump_json(indent=2)}",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": url},
                    },
                    {
                        "type": "text",
                        "text": """
                            First take a moment to reason about the best set of headers for the tables.
                            Write a good h1 for the image above. Then follow up with a short description of the what the data is about.
                            Then for each table you identified, write a h2 tag that is a descriptive title of the table.
                            Then follow up with a short description of the what the data is about.
                            Lastly, produce the markdown table for each table you identified.
                            Make sure to escape the markdown table properly, and make sure to include the caption and the dataframe.
                            including escaping all the newlines and quotes. Only return a markdown table in dataframe, nothing else.
                        """,
                    },
                ],
            }
        ],
    )
    return tables


if __name__ == "__main__":
    urls = [
        "https://a.storyblok.com/f/47007/2400x2000/bf383abc3c/231031_uk-ireland-in-three-charts_table_v01_b.png/m/2880x0",
    ]
    for url in urls:
        tables = extract(url)
        for table in tables.tables:
            print(table.caption)
            #> Top 10 Grossing Android Apps
            """
                        App Name                    Category
             Rank
            1                           Google One       Productivity
            2                              Disney+      Entertainment
            3        TikTok - Videos, Music & LIVE      Entertainment
            4                     Candy Crush Saga              Games
            5       Tinder: Dating, Chat & Friends  Social networking
            6                          Coin Master              Games
            7                               Roblox              Games
            8       Bumble - Dating & Make Friends             Dating
            9                          Royal Match              Games
            10         Spotify: Music and Podcasts      Music & Audio
            """
            print(table.dataframe)
            """
                        App Name                    Category
             Rank
            1       Tinder: Dating, Chat & Friends  Social networking
            2                              Disney+      Entertainment
            3       YouTube: Watch, Listen, Stream      Entertainment
            4         Audible: Audio Entertainment      Entertainment
            5                     Candy Crush Saga              Games
            6        TikTok - Videos, Music & LIVE      Entertainment
            7       Bumble - Dating & Make Friends             Dating
            8                               Roblox              Games
            9          LinkedIn: Job Search & News           Business
            10         Duolingo - Language Lessons          Education
            """
```
