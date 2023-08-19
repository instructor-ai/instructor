from datetime import datetime
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    ForeignKey,
    DateTime,
)
from sqlalchemy.orm import declarative_base, relationship, Session

Base = declarative_base()


class Message(Base):
    __tablename__ = "message"
    id = Column(Integer, primary_key=True, index=True)
    index = Column(Integer)
    role = Column(String)
    content = Column(String, index=True)
    arguments = Column(String)

    chatcompletion_id = Column(String, ForeignKey("chatcompletion.id"))
    chatcompletion = relationship(
        "ChatCompletion", back_populates="messages", foreign_keys=[chatcompletion_id]
    )

    response_chatcompletion_id = Column(String, ForeignKey("chatcompletion.id"))
    response_chatcompletion = relationship(
        "ChatCompletion",
        back_populates="responses",
        foreign_keys=[response_chatcompletion_id],
    )


class ChatCompletion(Base):
    __tablename__ = "chatcompletion"
    id = Column(String, primary_key=True)

    messages = relationship(
        "Message",
        back_populates="chatcompletion",
        foreign_keys=[Message.chatcompletion_id],
    )
    responses = relationship(
        "Message",
        back_populates="response_chatcompletion",
        foreign_keys=[Message.response_chatcompletion_id],
    )

    created_at = Column(DateTime, default=datetime.utcnow)
    temperature = Column(Float)
    model = Column(String)
    max_tokens = Column(Integer)
    prompt_tokens = Column(Integer)
    completion_tokens = Column(Integer)
    total_tokens = Column(Integer)


if __name__ == "__main__":
    sqlite_file_name = "chat.db"
    sqlite_url = f"sqlite:///{sqlite_file_name}"

    engine = create_engine(sqlite_url, echo=True)
    # create all the tables 
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        messages = [
            Message(
                role="system",
                content="You are helping a customer with a technical issue.",
            ),
            Message(
                role="user", content="Hello, I am having trouble with my computer."
            ),
        ]
        responses = [
            Message(
                role="system",
                content="Ok, what is the problem?",
            ),
            Message(role="user", content="My computer is not turning on."),
        ]
        chat_completion = ChatCompletion(
            id="test2",
            messages=messages,
            responses=responses,
            temperature=0.1,
            model="gpt-3.5-turbo-0613",
            max_tokens=1000,
            prompt_tokens=100,
            completion_tokens=900,
            total_tokens=1000,
        )
        session.add(chat_completion)
        session.commit()
