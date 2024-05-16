# Extracting Tables using GPT-Vision

This post demonstrates how to use Python's type annotations and OpenAI's new vision model to extract tables from images and convert them into markdown format. This method is particularly useful for data analysis and automation tasks.

The full code is available on [GitHub](https://github.com/jxnl/instructor/blob/main/examples/vision/run_table.py)

## Building the Custom Type for Markdown Tables

First, we define a custom type, `MarkdownDataFrame`, to handle pandas DataFrames formatted in markdown. This type uses Python's `Annotated` and `InstanceOf` types, along with decorators `BeforeValidator` and `PlainSerializer`, to process and serialize the data.

```python
from io import StringIO
from typing import Annotated, Any
from pydantic import BaseModel, Field, BeforeValidator, PlainSerializer, InstanceOf, WithJsonSchema
from typing import Iterable
import pandas as pd


def md_to_df(data: Any) -> Any:
    # Convert markdown to DataFrame
    if isinstance(data, str):
        return (
            pd.read_csv(
                StringIO(data),  # Process data
                sep="|",
                index_col=1,
            )
            .dropna(axis=1, how="all")
            .iloc[1:]
            .applymap(lambda x: x.strip())
        )
    return data


MarkdownDataFrame = Annotated[
    InstanceOf[pd.DataFrame],
    BeforeValidator(md_to_df),
    PlainSerializer(lambda df: df.to_markdown()),
    WithJsonSchema(
        {
            "type": "string",
            "description": "The markdown representation of the table, each one should be tidy, do not try to join tables that should be seperate",
        }
    ),
]
```

## Defining the Table Class

The `Table` class is essential for organizing the extracted data. It includes a caption and a dataframe, processed as a markdown table. Since most of the complexity is handled by the `MarkdownDataFrame` type, the `Table` class is straightforward!

```python
class Table(BaseModel):
    caption: str
    dataframe: MarkdownDataFrame
```

## Extracting Tables from Images

The `extract_table` function uses OpenAI's vision model to process an image URL and extract tables in markdown format. We utilize the `instructor` library to patch the OpenAI client for this purpose.

```python
import instructor
from openai import OpenAI

# Apply the patch to the OpenAI client to support response_model
# Also use MD_JSON mode since the visino model does not support any special structured output mode
client = instructor.from_openai(OpenAI(), mode=instructor.function_calls.Mode.MD_JSON)


def extract_table(url: str) -> Iterable[Table]:
    return client.chat.completions.create(
        model="gpt-4-vision-preview",
        response_model=Iterable[Table],
        max_tokens=1800,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract table from image."},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }
        ],
    )
```

## Practical Example

In this example, we apply the method to extract data from an image showing the top grossing apps in Ireland for October 2023.

```python
url = "https://a.storyblok.com/f/47007/2400x2000/bf383abc3c/231031_uk-ireland-in-three-charts_table_v01_b.png"
tables = extract_table(url)
for table in tables:
    print(table.caption, end="\n")
    print(table.dataframe)
```

??? Note "Expand to see the output"

    ![Top 10 Grossing Apps in October 2023 for Ireland](https://a.storyblok.com/f/47007/2400x2000/bf383abc3c/231031_uk-ireland-in-three-charts_table_v01_b.png)

    ### Top 10 Grossing Apps in October 2023 (Ireland) for Android Platforms

    | Rank | App Name                         | Category           |
    |------|----------------------------------|--------------------|
    | 1    | Google One                       | Productivity       |
    | 2    | Disney+                          | Entertainment      |
    | 3    | TikTok - Videos, Music & LIVE    | Entertainment      |
    | 4    | Candy Crush Saga                 | Games              |
    | 5    | Tinder: Dating, Chat & Friends   | Social networking  |
    | 6    | Coin Master                      | Games              |
    | 7    | Roblox                           | Games              |
    | 8    | Bumble - Dating & Make Friends   | Dating             |
    | 9    | Royal Match                      | Games              |
    | 10   | Spotify: Music and Podcasts      | Music & Audio      |

    ### Top 10 Grossing Apps in October 2023 (Ireland) for iOS Platforms

    | Rank | App Name                         | Category           |
    |------|----------------------------------|--------------------|
    | 1    | Tinder: Dating, Chat & Friends   | Social networking  |
    | 2    | Disney+                          | Entertainment      |
    | 3    | YouTube: Watch, Listen, Stream   | Entertainment      |
    | 4    | Audible: Audio Entertainment     | Entertainment      |
    | 5    | Candy Crush Saga                 | Games              |
    | 6    | TikTok - Videos, Music & LIVE    | Entertainment      |
    | 7    | Bumble - Dating & Make Friends   | Dating             |
    | 8    | Roblox                           | Games              |
    | 9    | LinkedIn: Job Search & News      | Business           |
    | 10   | Duolingo - Language Lessons      | Education          |
