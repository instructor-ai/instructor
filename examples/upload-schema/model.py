from pydantic import BaseModel
from enum import Enum


class Topic(Enum):
    SCIENCE = "science"
    TECHNOLOGY = "technology"
    ART = "art"


class Article(BaseModel):
    title: str
    content: str
    topic: Topic
