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


client = instructor.patch(OpenAI())


tables = client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=Iterable[Table],
    messages=[
        {
            "role": "system",
            "content": "Please extract the tables from the following text, merge as much as possible:",
        },
        {
            "role": "user",
            "content": """
        My name is John and I am 25 years old. I live in 
        New York and I like to play basketball. His name is 
        Mike and he is 30 years old. He lives in San Francisco 
        and he likes to play baseball. Sarah is 20 years old 
        and she lives in Los Angeles. She likes to play tennis.
        Her name is Mary and she is 35 years old. 
        She lives in Chicago.
        """,
        },
    ],
)

for table in tables:
    print(table.caption)
    print(table.dataframe)
    print()
    """
    People
            Age           City       Hobby 
    Name                                   
    John      25       New York  Basketball
    Mike      30  San Francisco    Baseball
    Sarah     20    Los Angeles      Tennis
    Mary      35        Chicago         N/A
    """
