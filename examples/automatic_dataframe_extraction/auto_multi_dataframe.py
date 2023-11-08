from instructor import OpenAISchema
from pydantic import Field
from typing import List, Any
from openai import OpenAI

client = OpenAI()


class RowData(OpenAISchema):
    row: List[Any] = Field(..., description="Correct values for each row")


class Dataframe(OpenAISchema):
    name: str = Field(..., description="The name of the dataframe")
    data: List[RowData] = Field(
        ...,
        description="Correct rows of data aligned to column names, Nones are allowed",
    )
    columns: List[str] = Field(
        ...,
        description="Column names relevant from source data, should be in snake_case",
    )

    def to_pandas(self):
        import pandas as pd

        columns = self.columns
        data = [row.row for row in self.data]

        return pd.DataFrame(data=data, columns=columns)


class Database(OpenAISchema):
    """
    A set of correct named and defined tables as dataframes
    Each one should have the right number of columns and correct
    values for each.
    """

    tables: List[Dataframe] = Field(
        ...,
        description="List of tables in the database",
    )


def dataframe(data: str) -> Database:
    completion = client.chat.completions.create(
        model="gpt-4-0613",
        temperature=0.0,
        functions=[Database.openai_schema],
        function_call={"name": Database.openai_schema["name"]},
        messages=[
            {
                "role": "system",
                "content": """Map this data into a dataframe a
            nd correctly define the correct columns and rows""",
            },
            {
                "role": "user",
                "content": f"{data}",
            },
        ],
        max_tokens=1000,
    )
    return Database.from_response(completion)


if __name__ == "__main__":
    dfs = dataframe(
        """My name is John and I am 25 years old. I live in 
        New York and I like to play basketball. His name is 
        Mike and he is 30 years old. He lives in San Francisco 
        and he likes to play baseball. Sarah is 20 years old 
        and she lives in Los Angeles. She likes to play tennis.
        Her name is Mary and she is 35 years old. 
        She lives in Chicago.

        On one team 'Tigers' the captan is John and there are 12 players.
        On the other team 'Lions' the captan is Mike and there are 10 players.
        """
    )

    for df in dfs.tables:
        print(df.name)
        print(df.to_pandas())
    """
    People
    ID   Name  Age           City Favorite Sport
    0   1   John   25       New York     Basketball
    1   2   Mike   30  San Francisco       Baseball
    2   3  Sarah   20    Los Angeles         Tennis
    3   4   Mary   35        Chicago           None
    Teams
    ID Team Name Captain  Number of Players
    0   1    Tigers    John                 12
    1   2     Lions    Mike                 10
    """
