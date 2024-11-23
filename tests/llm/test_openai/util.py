import instructor
from typing import List, Literal

models: List[str] = ["gpt-4-turbo-preview"]
modes: List[instructor.Mode] = [
    instructor.Mode.TOOLS,
    instructor.Mode.TOOLS_STRICT,
]
