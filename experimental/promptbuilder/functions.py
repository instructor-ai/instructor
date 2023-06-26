from pydantic import Field
from openai_function_call import OpenAISchema
from enum import Enum, auto


class Source(Enum):
    VIDEO = auto()
    DOCUMENT = auto()
    EMAIL = auto()
    OTHER = auto()


class Search(OpenAISchema):
    query: str
    name: str
    source: Source
