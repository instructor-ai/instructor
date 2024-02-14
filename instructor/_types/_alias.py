from typing import Literal

from typing_extensions import TypeAlias

ModelNames: TypeAlias = Literal[
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-instruct",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-4-0125-preview",
    "gpt-4-1106-preview",
    "gpt-4-1106-vision-preview",
    "gpt-4",
    "gpt-4-32k",
    "text-embedding-3-small",
    "text-embedding-3-large",
    "ada v2",
    "text-embedding-ada-002",
    "text-embedding-ada-002-v2",
]
