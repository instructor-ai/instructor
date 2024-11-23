"""Type stubs for instructor.mode."""
from enum import Enum
from typing import Set

class Mode(Enum):
    JSON = "json"
    MD_JSON = "md_json"
    TOOLS = "tools"
    TOOLS_STRICT = "tools_strict"
    FUNCTIONS = "functions"
    JSON_SCHEMA = "json_schema"
    WRITER_TOOLS = "writer_tools"
    ANTHROPIC_JSON = "anthropic_json"
    ANTHROPIC_TOOLS = "anthropic_tools"
    GEMINI_JSON = "gemini_json"
    GEMINI_TOOLS = "gemini_tools"
    CEREBRAS_JSON = "cerebras_json"
    FIREWORKS_JSON = "fireworks_json"
    FIREWORKS_TOOLS = "fireworks_tools"

    @staticmethod
    def warn_mode_functions_deprecation() -> None: ...

    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
