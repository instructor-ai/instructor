"""
Utility configurations for core provider tests.

This consolidates model/mode configuration for all core providers.
"""

import instructor

# OpenAI configurations
OPENAI_MODELS = ["openai/gpt-5-nano"]
OPENAI_MODES = [instructor.Mode.TOOLS]

# Anthropic configurations
ANTHROPIC_MODELS = ["anthropic/claude-3-7-sonnet-latest"]
ANTHROPIC_MODES = [instructor.Mode.ANTHROPIC_TOOLS]

# Google configurations (both genai and gemini use the same models)
GOOGLE_MODELS = ["google/gemini-2.0-flash-exp"]
GOOGLE_MODES = [instructor.Mode.GENAI_TOOLS, instructor.Mode.GENAI_STRUCTURED_OUTPUTS]

# Mistral configurations (for standalone tests if needed)
MISTRAL_MODELS = ["mistral/ministral-8b-latest"]
MISTRAL_MODES = [
    instructor.Mode.MISTRAL_STRUCTURED_OUTPUTS,
    instructor.Mode.MISTRAL_TOOLS,
]
