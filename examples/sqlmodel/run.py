import instructor
from openai import OpenAI
from typing import Optional
from sqlmodel import Field, SQLModel, create_engine, Session


# Define the model that will serve as a Table for the database
class Hero(SQLModel, instructor.OpenAISchema, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    secret_name: str
    age: Optional[int] = None


# Function to query OpenAI for a Hero record
client = instructor.patch(OpenAI())


def create_hero() -> Hero:
    return client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=Hero,
        messages=[
            {"role": "user", "content": "Make a new superhero"},
        ],
    )


# Insert the response into the database
engine = create_engine("sqlite:///database.db")
SQLModel.metadata.create_all(engine)

hero = create_hero()
print(hero.model_dump())


with Session(engine) as session:
    session.add(hero)
    session.commit()
