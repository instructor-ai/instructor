from typing import List, Optional
from datetime import datetime
from sqlmodel import Field, Relationship, Session, SQLModel, create_engine


class Message(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    index: Optional[int] = Field(default=None)
    role: str
    content: Optional[str] = Field(default=None, index=True)
    arguments: Optional[str] = None

    chatcompletion_id: Optional[str] = Column(
        default=None, foreign_key="chatcompletion.id"
    chatcompletion: Optional["ChatCompletion"] = Relationship(
        back_populates="messages",
        sa_relationship_kwargs={"foreign_keys": ["chatcompletion_id"]},
    )

    response_chatcompletion_id: Optional[str] = Field(
        default=None, foreign_key="chatcompletion.id"
    )
    response_chatcompletion: Optional["ChatCompletion"] = Relationship(
        back_populates="responses",
        sa_relationship_kwargs={"foreign_keys": ["response_chatcompletion_id"]},
    )


class ChatCompletion(SQLModel, table=True):
    id: Optional[str] = Field(default=None, primary_key=True)

    messages: List["Message"] = Relationship(back_populates="chatcompletion")
    responses: List["Message"] = Relationship(back_populates="response_chatcompletion")

    finish_reason: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    temperature: Optional[float] = None
    model: Optional[str]
    max_tokens: Optional[int] = None

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


if __name__ == "__main__":
    sqlite_file_name = "example.db"
    sqlite_url = f"sqlite:///{sqlite_file_name}"

    engine = create_engine(sqlite_url, echo=True)
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        chat_completion = ChatCompletion(
            messages=[
                Message(
                    role="system",
                    content="You are helping a customer with a technical issue.",
                ),
                Message(
                    role="user", content="Hello, I am having trouble with my computer."
                ),
                Message(role="system", content="What is the problem?"),
            ],
            responses=[Message(role="user", content="It is not working.")],
            temperature=0.1,
            model="gpt-3.5-turbo-0613",
            max_tokens=1000,
            prompt_tokens=12,
            completion_tokens=13,
        )
        session.add(chat_completion)

        chat_completion = ChatCompletion(
            messages=[
                Message(
                    role="system",
                    content="You are helping a customer with a technical issue.",
                ),
                Message(
                    role="user",
                    content="Hello, there I am having trouble with my computer.",
                ),
                Message(role="system", content="What is the problem?"),
            ],
            temperature=0.1,
            model="gpt-3.5-turbo-0613",
            max_tokens=1000,
            prompt_tokens=12,
            completion_tokens=13,
        )
        session.add(chat_completion)
        session.commit()
