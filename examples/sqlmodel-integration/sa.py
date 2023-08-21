from datetime import datetime
from sqlalchemy import (
    Boolean,
    create_engine,
    Column,
    Integer,
    String,
    Float,
    ForeignKey,
    DateTime,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Message(Base):
    __tablename__ = "message"
    id = Column(Integer, primary_key=True, index=True)
    index = Column(Integer)
    role = Column(String)
    content = Column(String, index=True)
    is_function_call = Column(Boolean)
    arguments = Column(String)
    name = Column(String)

    is_response = Column(Boolean, default=False)
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
    functions = Column(String)  # TODO: make this a foreign key
    function_call = Column(String)  # TODO: make this a foreign key


if __name__ == "__main__":
    sqlite_file_name = "openai.db"
    sqlite_url = f"sqlite:///{sqlite_file_name}"

    engine = create_engine(sqlite_url, echo=True)
    # create all the tables
    Base.metadata.create_all(engine)
