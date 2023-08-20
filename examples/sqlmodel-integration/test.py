import openai

from patch_sql import instrument_with_sqlalchemy
from instructor.patch import patch_chatcompletion_with_response_model

from typing import Optional
from sqlmodel import SQLModel, Session, Field

from sqlalchemy import create_engine

# SQLAlchemy allows you to support any database!
engine = create_engine("sqlite:///openai.db", echo=True)

# This saves all completions and messages into the database
# Allowing you to own your entire LLM observability (and data)
instrument_with_sqlalchemy(engine)

# This adds a response_model parameter to the ChatCompletion.careate method
patch_chatcompletion_with_response_model()


# Instead of Pydantic, you can use SQLModel!
class UserDetails(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    age: int


resp: UserDetails = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    response_model=UserDetails,
    messages=[
        {
            "role": "user",
            "content": "Extract jason is 25 years old",
        }
    ],
)  # type: ignore

# Hopefully this can be automatically done in the future
with Session(engine) as session:
    session.add(resp)
    session.commit()
