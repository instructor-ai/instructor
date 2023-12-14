from typing import Literal

from typing_extensions import TypeAlias

ModelNames: TypeAlias = Literal[
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-4",
    "gpt-4-32k",
    "text-embedding-ada-002",
    "text-embedding-ada-002-v2",
]
