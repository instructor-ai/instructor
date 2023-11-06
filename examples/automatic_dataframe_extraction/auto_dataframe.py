from instructor import OpenAISchema
from pydantic import Field
from typing import List, Any
from openai import OpenAI

client = OpenAI()


class RowData(OpenAISchema):
    row: List[Any] = Field(..., description="The values for each row")


class Dataframe(OpenAISchema):
    """
    Class representing a dataframe. This class is used to convert
    data into a frame that can be used by pandas.
    """

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


def dataframe(data: str) -> Dataframe:
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-0613",
        temperature=0.1,
        functions=[Dataframe.openai_schema],
        function_call={"name": Dataframe.openai_schema["name"]},
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
    return Dataframe.from_response(completion)


if __name__ == "__main__":
    df = dataframe(
        """My name is John and I am 25 years old. I live in 
        New York and I like to play basketball. His name is 
        Mike and he is 30 years old. He lives in San Francisco 
        and he likes to play baseball. Sarah is 20 years old 
        and she lives in Los Angeles. She likes to play tennis.
        Her name is Mary and she is 35 years old. 
        She lives in Chicago."""
    )

    print(df.to_pandas())
    """
        name  age       location       hobby
    0   John   25       New York  basketball
    1   Mike   30  San Francisco    baseball
    2  Sarah   20    Los Angeles      tennis
    3   Mary   35        Chicago        None
    """
