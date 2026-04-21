"""Minimal MiniMax + Instructor example.

Set ``MINIMAX_API_KEY`` before running. MiniMax uses an OpenAI-compatible
chat completions endpoint, so Instructor wires it up with ``from_minimax``.
"""

from __future__ import annotations

import instructor
from pydantic import BaseModel


class UserInfo(BaseModel):
    name: str
    age: int


def main() -> None:
    client = instructor.from_minimax()

    user = client.chat.completions.create(
        model="MiniMax-Text-01",
        messages=[{"role": "user", "content": "Extract: John is 25 years old."}],
        response_model=UserInfo,
    )

    print(user)


if __name__ == "__main__":
    main()
