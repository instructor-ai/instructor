import enum
import warnings


# Track if deprecation warning has been shown
_functions_deprecation_shown = False


class Mode(enum.Enum):
    """
    Mode enumeration for patching LLM API clients.

    Each mode determines how the library formats and structures requests
    to different provider APIs and how it processes their responses.
    """

    # OpenAI modes
    FUNCTIONS = "function_call"  # Deprecated
    PARALLEL_TOOLS = "parallel_tool_call"
    TOOLS = "tool_call"
    TOOLS_STRICT = "tools_strict"
    JSON = "json_mode"
    JSON_O1 = "json_o1"
    MD_JSON = "markdown_json_mode"
    JSON_SCHEMA = "json_schema_mode"

    # Add new modes to support responses api
    RESPONSES_TOOLS = "responses_tools"
    RESPONSES_TOOLS_WITH_INBUILT_TOOLS = "responses_tools_with_inbuilt_tools"

    # Anthropic modes
    ANTHROPIC_TOOLS = "anthropic_tools"
    ANTHROPIC_REASONING_TOOLS = "anthropic_reasoning_tools"
    ANTHROPIC_JSON = "anthropic_json"
    ANTHROPIC_PARALLEL_TOOLS = "anthropic_parallel_tools"

    # Mistral modes
    MISTRAL_TOOLS = "mistral_tools"
    MISTRAL_STRUCTURED_OUTPUTS = "mistral_structured_outputs"

    # Vertex AI & Google modes
    VERTEXAI_TOOLS = "vertexai_tools"
    VERTEXAI_JSON = "vertexai_json"
    VERTEXAI_PARALLEL_TOOLS = "vertexai_parallel_tools"
    GEMINI_JSON = "gemini_json"
    GEMINI_TOOLS = "gemini_tools"
    GENAI_TOOLS = "genai_tools"
    GENAI_STRUCTURED_OUTPUTS = "genai_structured_outputs"

    # Cohere modes
    COHERE_TOOLS = "cohere_tools"
    COHERE_JSON_SCHEMA = "json_object"

    # Cerebras modes
    CEREBRAS_TOOLS = "cerebras_tools"
    CEREBRAS_JSON = "cerebras_json"

    # Fireworks modes
    FIREWORKS_TOOLS = "fireworks_tools"
    FIREWORKS_JSON = "fireworks_json"

    # Other providers
    WRITER_TOOLS = "writer_tools"
    WRITER_JSON = "writer_json"
    BEDROCK_TOOLS = "bedrock_tools"
    BEDROCK_JSON = "bedrock_json"
    PERPLEXITY_JSON = "perplexity_json"
    OPENROUTER_STRUCTURED_OUTPUTS = "openrouter_structured_outputs"

    # Classification helpers
    @classmethod
    def tool_modes(cls) -> set["Mode"]:
        """Returns a set of all tool-based modes."""
        return {
            cls.FUNCTIONS,
            cls.PARALLEL_TOOLS,
            cls.TOOLS,
            cls.TOOLS_STRICT,
            cls.ANTHROPIC_TOOLS,
            cls.ANTHROPIC_REASONING_TOOLS,
            cls.ANTHROPIC_PARALLEL_TOOLS,
            cls.MISTRAL_TOOLS,
            cls.VERTEXAI_TOOLS,
            cls.VERTEXAI_PARALLEL_TOOLS,
            cls.GEMINI_TOOLS,
            cls.COHERE_TOOLS,
            cls.CEREBRAS_TOOLS,
            cls.FIREWORKS_TOOLS,
            cls.WRITER_TOOLS,
            cls.BEDROCK_TOOLS,
            cls.OPENROUTER_STRUCTURED_OUTPUTS,
            cls.MISTRAL_STRUCTURED_OUTPUTS,
        }

    @classmethod
    def json_modes(cls) -> set["Mode"]:
        """Returns a set of all JSON-based modes."""
        return {
            cls.JSON,
            cls.JSON_O1,
            cls.MD_JSON,
            cls.JSON_SCHEMA,
            cls.ANTHROPIC_JSON,
            cls.VERTEXAI_JSON,
            cls.GEMINI_JSON,
            cls.COHERE_JSON_SCHEMA,
            cls.CEREBRAS_JSON,
            cls.FIREWORKS_JSON,
            cls.WRITER_JSON,
            cls.BEDROCK_JSON,
            cls.PERPLEXITY_JSON,
            cls.OPENROUTER_STRUCTURED_OUTPUTS,
            cls.MISTRAL_STRUCTURED_OUTPUTS,
        }

    @classmethod
    def warn_mode_functions_deprecation(cls):
        """
        Warn about FUNCTIONS mode deprecation.

        Shows the warning only once per session to avoid spamming logs
        with the same message.
        """
        global _functions_deprecation_shown
        if not _functions_deprecation_shown:
            warnings.warn(
                "The FUNCTIONS mode is deprecated and will be removed in future versions",
                DeprecationWarning,
                stacklevel=2,
            )
            _functions_deprecation_shown = True
