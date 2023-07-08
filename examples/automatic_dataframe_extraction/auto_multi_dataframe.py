from openai_function_call import OpenAISchema
from pydantic import Field
from typing import List, Any
import openai


class RowData(OpenAISchema):
    row: List[Any] = Field(..., description="The values for each row")
    citation: str = Field(
        ..., description="The citation for this row from the original source data"
    )


class Dataframe(OpenAISchema):
    """
    Class representing a dataframe. This class is used to convert
    data into a frame that can be used by pandas.
    """

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

        columns = self.columns + ["citation"]
        data = [row.row + [row.citation] for row in self.data]

        return pd.DataFrame(data=data, columns=columns)


class Database(OpenAISchema):
    """
    A set of correct named and defined tables as dataframes
    """

    tables: List[Dataframe] = Field(
        ...,
        description="List of tables in the database",
    )


def dataframe(data: str) -> Database:
    completion = openai.ChatCompletion.create(
        model="gpt-4-0613",
        temperature=0.1,
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
    Name  Age           City Favorite Sport
    0   John   25       New York     Basketball
    1   Mike   30  San Francisco       Baseball
    2  Sarah   20    Los Angeles         Tennis
    3   Mary   35        Chicago           None

    Teams
    Team Name Captain  Number of Players
    0    Tigers    John                 12
    1     Lions    Mike                 10
    """
