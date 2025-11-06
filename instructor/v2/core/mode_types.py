"""Hierarchical mode type definitions for v2.

Instead of 42 flat mode enums (ANTHROPIC_TOOLS, GEMINI_JSON, etc.),
v2 uses a hierarchical (Provider, ModeType) tuple system:

- 15 Provider values (ANTHROPIC, OPENAI, GEMINI, ...)
- 6 ModeType values (TOOLS, JSON, PARALLEL_TOOLS, ...)
- Any combination: (Provider.ANTHROPIC, ModeType.TOOLS)

Benefits:
- Composable: 21 enums define any combination vs 42 hardcoded
- Queryable: Filter by provider OR mode type
- Extensible: Add provider = support all mode types
- Clear semantics: (ANTHROPIC, TOOLS) vs ANTHROPIC_TOOLS
"""

from enum import Enum


class Provider(Enum):
    """Supported LLM providers.

    Each provider can support multiple mode types.
    """

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"
    COHERE = "cohere"
    MISTRAL = "mistral"
    GROQ = "groq"
    VERTEXAI = "vertexai"
    FIREWORKS = "fireworks"
    CEREBRAS = "cerebras"
    WRITER = "writer"
    DATABRICKS = "databricks"
    ANYSCALE = "anyscale"
    TOGETHER = "together"
    LITELLM = "litellm"
    BEDROCK = "bedrock"
    PERPLEXITY = "perplexity"
    XAI = "xai"


class ModeType(Enum):
    """Types of structured output modes.

    Each mode type defines how structured data is extracted from LLM responses.
    """

    # Function/tool calling modes
    TOOLS = "tools"
    PARALLEL_TOOLS = "parallel_tools"
    REASONING_TOOLS = "reasoning_tools"

    # JSON modes
    JSON = "json"
    JSON_SCHEMA = "json_schema"
    STRUCTURED_OUTPUTS = "structured_outputs"

    # Special modes
    MD_JSON = "md_json"  # Markdown-wrapped JSON


# Type alias for mode tuples
Mode = tuple[Provider, ModeType]
