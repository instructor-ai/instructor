from typing import List, Optional

from sqlmodel import Field, Relationship, Session, SQLModel, create_engine


class Messages(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    role: str
    content: Optional[str] = None
    arguments: Optional[str] = None

    completion_id: Optional[int] = Field(default=None, foreign_key="completion.id")
    completion: Optional["Completion"] = Relationship(back_populates="messages")


class Completion(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)

    messages: List["Messages"] = Relationship(back_populates="completion")
    temperature: Optional[float] = None
    model: Optional[str]
    max_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

    queries: List["Query"] = Relationship(back_populates="completion")


class Team(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    headquarters: str

    heroes: List["Hero"] = Relationship(back_populates="team")


class Query(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: Optional[str] = None
    subqueries: List["SubQuery"] = Relationship(back_populates="query")

    completion_id: Optional[int] = Field(default=None, foreign_key="completion.id")
    completion: Optional[Completion] = Relationship(back_populates="queries")


class Hero(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    secret_name: str
    age: Optional[int] = Field(default=None, index=True)

    team_id: Optional[int] = Field(default=None, foreign_key="team.id")
    team: Optional[Team] = Relationship(back_populates="heroes")


class SubQuery(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: Optional[str] = None

    query_id: Optional[int] = Field(default=None, foreign_key="query.id")
    query: Optional[Query] = Relationship(back_populates="subqueries")


sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

engine = create_engine(sqlite_url, echo=True)


SQLModel.metadata.create_all(engine)
