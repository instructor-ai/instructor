import openai

from patch_sql import instrument_chat_completion_sa, instrument_with_sa_engine
from instructor.patch import patch_chatcompletion_with_response_model

from typing import Optional
from sqlmodel import SQLModel, Session, Field

from sqlalchemy import create_engine

# SQLAlchemy allows you to support any database!
engine = create_engine("sqlite:///openai.db", echo=True)

instrument_with_sa_engine(engine)

# This adds a response_model parameter to the ChatCompletion.careate method
patch_chatcompletion_with_response_model()


# Instead of Pydantic, you can use SQLModel!
class UserDetails(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    completion_id: Optional[str] = Field(default=None)
    name: str
    age: int

    def save(self):
        with Session(engine) as session:
            
            if hasattr(self, "_raw_response"):
                self.completion_id = self._raw_response["id"]

            session.add(self)
            session.commit()


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

resp.save()
