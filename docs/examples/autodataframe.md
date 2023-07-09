# Example: Converting Text into Dataframes

In this example, we'll demonstrate how to convert a text into dataframes using OpenAI Function Call. We will define the necessary data structures using Pydantic and show how to convert the text into dataframes.

## Defining the Data Structures

Let's start by defining the data structures required for this task: `RowData`, `Dataframe`, and `Database`.

```python
from openai_function_call import OpenAISchema
from pydantic import Field
from typing import List, Any


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
```

The `RowData` class represents a single row of data in the dataframe. It contains a `row` attribute for the values in each row and a `citation` attribute for the citation from the original source data.

The `Dataframe` class represents a dataframe and consists of a `name` attribute, a list of `RowData` objects in the `data` attribute, and a list of column names in the `columns` attribute. It also provides a `to_pandas` method to convert the dataframe into a Pandas DataFrame.

The `Database` class represents a set of tables in a database. It contains a list of `Dataframe` objects in the `tables` attribute.

## Using the Prompt Pipeline

To convert a text into dataframes, we'll use the Prompt Pipeline in OpenAI Function Call. We can define a function `dataframe` that takes a text as input and returns a `Database` object.

```python
import openai

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
```

The `dataframe` function takes a string `data` as input and creates a completion using the Prompt Pipeline. It prompts the model to map the data into a dataframe and define the correct columns and rows. The resulting completion is then converted into a `Database` object.

## Evaluating an Example

Let's evaluate the example by converting a text into dataframes using the `dataframe` function and print the resulting dataframes.

```python
dfs = dataframe("""My name is John and I am 25 years old. I live in 
New York and I like to play basketball. His name is 
Mike and he is 30 years old. He lives in San Francisco 
and he likes to play baseball. Sarah is 20 years old 
and she lives in Los Angeles. She likes to play tennis.
Her name is Mary and she is 35 years old. 
She lives in Chicago.

On one team 'Tigers' the captain is John and there are 12 players.
On the other team 'Lions' the captain is Mike and there are 10 players.
""")

for df in dfs.tables:
    print(df.name)
    print(df.to_pandas())
```

The output will be:

```sh
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
```
