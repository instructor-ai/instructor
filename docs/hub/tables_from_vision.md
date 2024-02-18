# Extracting Tables from Images with OpenAI's GPT-4 Vision Model

First, we define a custom type, `MarkdownDataFrame`, to handle pandas DataFrames formatted in markdown. This type uses Python's `Annotated` and `InstanceOf` types, along with decorators `BeforeValidator` and `PlainSerializer`, to process and serialize the data.

## Defining the Table Class

The `Table` class is essential for organizing the extracted data. It includes a caption and a dataframe, processed as a markdown table. Since most of the complexity is handled by the `MarkdownDataFrame` type, the `Table` class is straightforward!

```python
from io import StringIO
from typing import Annotated, Any
from pydantic import (
    BaseModel,
    BeforeValidator,
    PlainSerializer,
    InstanceOf,
    WithJsonSchema,
)
from typing import Iterable

import pandas as pd
import instructor
import openai


def md_to_df(data: Any) -> Any:
    """
    Convert markdown to pandas DataFrame
    """
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


# Define the MarkdownDataFrame type, a powerful type that can
# handle pandas DataFrames formatted in markdown
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


class Table(BaseModel):
    caption: str
    dataframe: MarkdownDataFrame


# Apply the patch to the OpenAI client to support response_model
# Also use MD_JSON mode since the visino model does not support any special structured output mode
client = instructor.patch(openai.OpenAI(), mode=instructor.function_calls.Mode.MD_JSON)


def extract_table(url: str) -> Iterable[Table]:
    """
    Extract tables from an image url using GPT-4 Vision

    Prompts it to output a table in markdown format,
    then we parse the markdown into a pandas DataFrame and return it
    """
    return client.chat.completions.create(
        model="gpt-4-vision-preview",
        response_model=Iterable[Table],
        max_tokens=1800,  # important to set this to a high value
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract table objects from image the image.",
                    },
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }
        ],
    )


if __name__ == "__main__":
    url = "https://a.storyblok.com/f/47007/2400x2000/bf383abc3c/231031_uk-ireland-in-three-charts_table_v01_b.png"
    tables = extract_table(url)
    for table in tables:
        print(table.caption, end="\n")
        print(table.dataframe)
```
