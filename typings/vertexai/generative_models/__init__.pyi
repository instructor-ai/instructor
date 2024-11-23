"""Type stubs for vertexai.generative_models."""
from typing import TypeAlias

class HarmCategory(str):
    HARM_CATEGORY_HATE_SPEECH: str = "HARM_CATEGORY_HATE_SPEECH"
    HARM_CATEGORY_HARASSMENT: str = "HARM_CATEGORY_HARASSMENT"
    HARM_CATEGORY_DANGEROUS_CONTENT: str = "HARM_CATEGORY_DANGEROUS_CONTENT"

class HarmBlockThreshold(str):
    BLOCK_ONLY_HIGH: str = "BLOCK_ONLY_HIGH"

SafetySettings: TypeAlias = dict[str, str]
