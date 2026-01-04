# type: ignore
import json
import logging
import re
from functools import wraps
from typing import Annotated, Any, Optional, TypeVar, cast
from openai.types.chat import ChatCompletion
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    create_model,
)

from ..core.exceptions import (
    IncompleteOutputException,
    ResponseParsingError,
    ConfigurationError,
)
from ..mode import Mode
from ..utils import (
    classproperty,
    extract_json_from_codeblock,
)
from .schema import (
    generate_openai_schema,
    generate_anthropic_schema,
    generate_gemini_schema,
)


T = TypeVar("T")
Model = TypeVar("Model", bound=BaseModel)

logger = logging.getLogger("instructor")

# No schema cache


# Utility functions for common JSON parsing operations
def _handle_incomplete_output(completion: Any) -> None:
    """Check if a completion was incomplete and raise appropriate exception."""
    if (
        hasattr(completion, "choices")
        and completion.choices[0].finish_reason == "length"
    ):
        raise IncompleteOutputException(last_completion=completion)

    # Handle Anthropic format
    if hasattr(completion, "stop_reason") and completion.stop_reason == "max_tokens":
        raise IncompleteOutputException(last_completion=completion)


def _extract_text_content(completion: Any) -> str:
    """Extract text content from various completion formats."""
    # OpenAI format
    if hasattr(completion, "choices"):
        return completion.choices[0].message.content or ""

    # Simple text format
    if hasattr(completion, "text"):
        return completion.text

    # Anthropic format
    if hasattr(completion, "content"):
        text_blocks = [c for c in completion.content if c.type == "text"]
        if text_blocks:
            return text_blocks[0].text

    # Bedrock format
    if isinstance(completion, dict) and "output" in completion:
        try:
            return completion.get("output").get("message").get("content")[0].get("text")
        except (AttributeError, IndexError):
            pass

    return ""


def _validate_model_from_json(
    cls: type[Any],
    json_str: str,
    validation_context: Optional[dict[str, Any]] = None,
    strict: Optional[bool] = None,
) -> Any:
    """Validate model from JSON string with appropriate error handling."""
    try:
        if hasattr(cls, "model_validate_json"):
            if strict:
                return cls.model_validate_json(
                    json_str, context=validation_context, strict=True
                )
            # Allow control characters
            parsed = json.loads(json_str, strict=False)
            return cls.model_validate(parsed, context=validation_context, strict=False)

        adapter = TypeAdapter(cls)
        if strict:
            return adapter.validate_json(
                json_str, context=validation_context, strict=True
            )
        parsed = json.loads(json_str, strict=False)
        return adapter.validate_python(parsed, context=validation_context, strict=False)
    except json.JSONDecodeError as e:
        logger.debug(f"JSON decode error: {e}")
        raise
    except Exception as e:
        logger.debug(f"Model validation error: {e}")
        raise


class OpenAISchema(BaseModel):
    # Ignore classproperty, since Pydantic doesn't understand it like it would a normal property.
    model_config = ConfigDict(ignored_types=(classproperty,))

    @classproperty
    def openai_schema(cls) -> dict[str, Any]:
        """
        Return the schema in the format of OpenAI's schema as jsonschema

        Note:
            Its important to add a docstring to describe how to best use this class, it will be included in the description attribute and be part of the prompt.

        Returns:
            model_json_schema (dict): A dictionary in the format of OpenAI's schema as jsonschema
        """
        return generate_openai_schema(cls)

    @classproperty
    def anthropic_schema(cls) -> dict[str, Any]:
        # Generate the Anthropic schema based on the OpenAI schema to avoid redundant schema generation
        return generate_anthropic_schema(cls)

    @classproperty
    def gemini_schema(cls) -> Any:
        # This is kept for backward compatibility but deprecated
        return generate_gemini_schema(cls)

    @classmethod
    def from_response(
        cls,
        completion: ChatCompletion,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
        mode: Mode = Mode.TOOLS,
    ) -> BaseModel:
        """Execute the function from the response of an openai chat completion

        Parameters:
            completion (openai.ChatCompletion): The response from an openai chat completion
            strict (bool): Whether to use strict json parsing
            mode (Mode): The openai completion mode

        Returns:
            cls (OpenAISchema): An instance of the class
        """

        if mode == Mode.ANTHROPIC_TOOLS:
            return cls.parse_anthropic_tools(completion, validation_context, strict)

        if mode == Mode.ANTHROPIC_TOOLS or mode == Mode.ANTHROPIC_REASONING_TOOLS:
            return cls.parse_anthropic_tools(completion, validation_context, strict)

        if mode == Mode.ANTHROPIC_JSON:
            return cls.parse_anthropic_json(completion, validation_context, strict)

        if mode == Mode.BEDROCK_JSON:
            return cls.parse_bedrock_json(completion, validation_context, strict)

        if mode == Mode.BEDROCK_TOOLS:
            return cls.parse_bedrock_tools(completion, validation_context, strict)

        if mode in {Mode.VERTEXAI_TOOLS, Mode.GEMINI_TOOLS}:
            return cls.parse_vertexai_tools(completion, validation_context)

        if mode == Mode.VERTEXAI_JSON:
            return cls.parse_vertexai_json(completion, validation_context, strict)

        if mode == Mode.COHERE_TOOLS:
            return cls.parse_cohere_tools(completion, validation_context, strict)

        if mode == Mode.GEMINI_JSON:
            return cls.parse_gemini_json(completion, validation_context, strict)

        if mode == Mode.GENAI_STRUCTURED_OUTPUTS:
            return cls.parse_genai_structured_outputs(
                completion, validation_context, strict
            )

        if mode == Mode.GEMINI_TOOLS:
            return cls.parse_gemini_tools(completion, validation_context, strict)

        if mode == Mode.GENAI_TOOLS:
            return cls.parse_genai_tools(completion, validation_context, strict)

        if mode == Mode.COHERE_JSON_SCHEMA:
            return cls.parse_cohere_json_schema(completion, validation_context, strict)

        if mode == Mode.WRITER_TOOLS:
            return cls.parse_writer_tools(completion, validation_context, strict)

        if mode == Mode.WRITER_JSON:
            return cls.parse_writer_json(completion, validation_context, strict)

        if mode in {Mode.RESPONSES_TOOLS, Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS}:
            return cls.parse_responses_tools(
                completion,
                validation_context,
                strict,
            )

        if not completion.choices:
            # This helps catch errors from OpenRouter
            if hasattr(completion, "error"):
                raise ResponseParsingError(
                    f"LLM provider returned error: {completion.error}",
                    mode=str(mode),
                    raw_response=completion,
                )

            raise ResponseParsingError(
                "No completion choices found in LLM response",
                mode=str(mode),
                raw_response=completion,
            )

        if completion.choices[0].finish_reason == "length":
            raise IncompleteOutputException(last_completion=completion)

        if mode == Mode.FUNCTIONS:
            Mode.warn_mode_functions_deprecation()
            return cls.parse_functions(completion, validation_context, strict)

        if mode == Mode.MISTRAL_STRUCTURED_OUTPUTS:
            return cls.parse_mistral_structured_outputs(
                completion, validation_context, strict
            )

        if mode in {
            Mode.TOOLS,
            Mode.MISTRAL_TOOLS,
            Mode.TOOLS_STRICT,
            Mode.CEREBRAS_TOOLS,
            Mode.FIREWORKS_TOOLS,
        }:
            return cls.parse_tools(completion, validation_context, strict)

        if mode in {
            Mode.JSON,
            Mode.JSON_SCHEMA,
            Mode.MD_JSON,
            Mode.JSON_O1,
            Mode.CEREBRAS_JSON,
            Mode.FIREWORKS_JSON,
            Mode.PERPLEXITY_JSON,
            Mode.OPENROUTER_STRUCTURED_OUTPUTS,
        }:
            return cls.parse_json(completion, validation_context, strict)

        raise ConfigurationError(
            f"Invalid or unsupported mode: {mode}. This mode may not be implemented for response parsing."
        )

    @classmethod
    def parse_genai_structured_outputs(
        cls: type[BaseModel],
        completion: ChatCompletion,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> BaseModel:
        return cls.model_validate_json(
            completion.text, context=validation_context, strict=strict
        )

    @classmethod
    def parse_genai_tools(
        cls: type[BaseModel],
        completion: ChatCompletion,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> BaseModel:
        from google.genai import types

        assert isinstance(completion, types.GenerateContentResponse)
        assert len(completion.candidates) == 1

        # Filter out thought parts (parts with thought: true)
        parts = completion.candidates[0].content.parts
        non_thought_parts = [
            part for part in parts if not (hasattr(part, "thought") and part.thought)
        ]

        assert len(non_thought_parts) == 1, (
            f"Instructor does not support multiple function calls, use List[Model] instead"
        )
        function_call = non_thought_parts[0].function_call
        assert function_call is not None, (
            f"Please return your response as a function call with the schema {cls.openai_schema} and the name {cls.openai_schema['name']}"
        )

        assert function_call.name == cls.openai_schema["name"]
        return cls.model_validate(
            obj=function_call.args, context=validation_context, strict=strict
        )

    @classmethod
    def parse_cohere_json_schema(
        cls: type[BaseModel],
        completion: ChatCompletion,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ):
        # Handle both V1 and V2 response structures
        if hasattr(completion, "text"):
            # V1 format: direct text access
            text = completion.text
        elif hasattr(completion, "message") and hasattr(completion.message, "content"):
            # V2 format: nested structure (message.content[].text)
            # V2 responses may have multiple content items (thinking, text, etc.)
            content_items = completion.message.content
            if content_items and len(content_items) > 0:
                # Find the text content item (skip thinking/other types)
                # TODO handle these other content types
                text = None
                for item in content_items:
                    if (
                        hasattr(item, "type")
                        and item.type == "text"
                        and hasattr(item, "text")
                    ):
                        text = item.text
                        break

                if text is None:
                    raise ResponseParsingError(
                        "Cohere V2 response has no text content item",
                        mode="COHERE_JSON_SCHEMA",
                        raw_response=completion,
                    )
            else:
                raise ResponseParsingError(
                    "Cohere V2 response has no content",
                    mode="COHERE_JSON_SCHEMA",
                    raw_response=completion,
                )
        else:
            raise ResponseParsingError(
                f"Unsupported Cohere response format. Expected 'text' (V1) or "
                f"'message.content[].text' (V2), got: {type(completion)}",
                mode="COHERE_JSON_SCHEMA",
                raw_response=completion,
            )

        return cls.model_validate_json(text, context=validation_context, strict=strict)

    @classmethod
    def parse_anthropic_tools(
        cls: type[BaseModel],
        completion: ChatCompletion,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> BaseModel:
        from anthropic.types import Message

        if isinstance(completion, Message) and completion.stop_reason == "max_tokens":
            raise IncompleteOutputException(last_completion=completion)

        # Anthropic returns arguments as a dict, dump to json for model validation below
        tool_calls = [
            json.dumps(c.input) for c in completion.content if c.type == "tool_use"
        ]  # TODO update with anthropic specific types

        tool_calls_validator = TypeAdapter(
            Annotated[list[Any], Field(min_length=1, max_length=1)]
        )
        tool_call = tool_calls_validator.validate_python(tool_calls)[0]

        return cls.model_validate_json(
            tool_call, context=validation_context, strict=strict
        )

    @classmethod
    def parse_anthropic_json(
        cls: type[BaseModel],
        completion: ChatCompletion,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> BaseModel:
        from anthropic.types import Message

        last_block = None

        if hasattr(completion, "choices"):
            completion = completion.choices[0]
            if completion.finish_reason == "length":
                raise IncompleteOutputException(last_completion=completion)
            text = completion.message.content
        else:
            assert isinstance(completion, Message)
            if completion.stop_reason == "max_tokens":
                raise IncompleteOutputException(last_completion=completion)
            # Find the last text block in the completion
            # this is because the completion is a list of blocks
            # and the last block is the one that contains the text ideally
            # this could happen due to things like multiple tool calls
            # read: https://docs.anthropic.com/en/docs/build-with-claude/tool-use/web-search-tool#response
            text_blocks = [c for c in completion.content if c.type == "text"]
            last_block = text_blocks[-1]
            text = last_block.text

        extra_text = extract_json_from_codeblock(text)

        if strict:
            model = cls.model_validate_json(
                extra_text, context=validation_context, strict=True
            )
        else:
            # Allow control characters to pass through by using the non-strict JSON parser.
            parsed = json.loads(extra_text, strict=False)
            # Pydantic non-strict: https://docs.pydantic.dev/latest/concepts/strict_mode/
            model = cls.model_validate(parsed, context=validation_context, strict=False)

        return model

    @classmethod
    def parse_bedrock_json(
        cls: type[BaseModel],
        completion: Any,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> BaseModel:
        if isinstance(completion, dict):
            # OpenAI will send the first content to be 'reasoningText', and then 'text'
            content = completion["output"]["message"]["content"]
            text_content = next((c for c in content if "text" in c), None)
            if not text_content:
                raise ResponseParsingError(
                    "Unexpected format. No text content found in Bedrock response.",
                    mode="BEDROCK_JSON",
                    raw_response=completion,
                )
            text = text_content["text"]
            match = re.search(r"```?json(.*?)```?", text, re.DOTALL)
            if match:
                text = match.group(1).strip()

            text = re.sub(r"```?json|\\n", "", text).strip()
        else:
            text = completion.text
        return cls.model_validate_json(text, context=validation_context, strict=strict)

    @classmethod
    def parse_bedrock_tools(
        cls: type[BaseModel],
        completion: Any,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> BaseModel:
        if isinstance(completion, dict):
            # Extract the tool use from Bedrock response
            message = completion.get("output", {}).get("message", {})
            content = message.get("content", [])

            # Find the tool use content block
            for content_block in content:
                if "toolUse" in content_block:
                    tool_use = content_block["toolUse"]
                    assert tool_use.get("name") == cls.__name__, (
                        f"Tool name mismatch: expected {cls.__name__}, got {tool_use.get('name')}"
                    )
                    return cls.model_validate(
                        tool_use.get("input", {}),
                        context=validation_context,
                        strict=strict,
                    )

            raise ResponseParsingError(
                "No tool use found in Bedrock response",
                mode="BEDROCK_TOOLS",
                raw_response=completion,
            )
        else:
            # Fallback for other response formats
            return cls.model_validate_json(
                completion.text, context=validation_context, strict=strict
            )

    @classmethod
    def parse_gemini_json(
        cls: type[BaseModel],
        completion: Any,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> BaseModel:
        try:
            text = completion.text
        except ValueError:
            logger.debug(
                f"Error response: {completion.result.candidates[0].finish_reason}\n\n{completion.result.candidates[0].safety_ratings}"
            )

        try:
            extra_text = extract_json_from_codeblock(text)  # type: ignore
        except UnboundLocalError:
            raise ResponseParsingError(
                "Unable to extract JSON from completion text. The response may have been blocked or empty.",
                mode="GEMINI_JSON",
                raw_response=completion,
            ) from None

        if strict:
            return cls.model_validate_json(
                extra_text, context=validation_context, strict=True
            )
        else:
            # Allow control characters.
            parsed = json.loads(extra_text, strict=False)
            # Pydantic non-strict: https://docs.pydantic.dev/latest/concepts/strict_mode/
            return cls.model_validate(parsed, context=validation_context, strict=False)

    @classmethod
    def parse_vertexai_tools(
        cls: type[BaseModel],
        completion: ChatCompletion,
        validation_context: Optional[dict[str, Any]] = None,
    ) -> BaseModel:
        tool_call = completion.candidates[0].content.parts[0].function_call.args  # type: ignore
        model = {}
        for field in tool_call:  # type: ignore
            model[field] = tool_call[field]
        # We enable strict=False because the conversion from protobuf -> dict often results in types like ints being cast to floats, as a result in order for model.validate to work we need to disable strict mode.
        return cls.model_validate(model, context=validation_context, strict=False)

    @classmethod
    def parse_vertexai_json(
        cls: type[BaseModel],
        completion: ChatCompletion,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> BaseModel:
        return cls.model_validate_json(
            completion.text, context=validation_context, strict=strict
        )

    @classmethod
    def parse_cohere_tools(
        cls: type[BaseModel],
        completion: ChatCompletion,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> BaseModel:
        """
        Parse Cohere tools response.

        Supports:
        - V1 native tool calls: completion.tool_calls[0].parameters
        - V2 native tool calls: completion.message.tool_calls[0].function.arguments (JSON string)
        - V1 text-based: completion.text (prompt-based approach)
        - V2 text-based: completion.message.content[].text (prompt-based approach)
        """
        # First, check for native Cohere tool calls (V1 and V2)
        # V1: completion.tool_calls with tc.parameters (dict)
        if hasattr(completion, "tool_calls") and completion.tool_calls:
            # V1 tool call format
            tool_call = completion.tool_calls[0]
            # Parameters in V1 are already a dict
            return cls.model_validate(
                tool_call.parameters, context=validation_context, strict=strict
            )

        # V2: completion.message.tool_calls with tc.function.arguments (JSON string)
        if (
            hasattr(completion, "message")
            and hasattr(completion.message, "tool_calls")
            and completion.message.tool_calls
        ):
            # V2 tool call format
            tool_call = completion.message.tool_calls[0]
            # Arguments in V2 are a JSON string
            import json

            arguments = json.loads(tool_call.function.arguments)
            return cls.model_validate(
                arguments, context=validation_context, strict=strict
            )

        # Fallback to text-based extraction (current prompt-based approach)
        # Handle both V1 and V2 text response structures
        if hasattr(completion, "text"):
            # V1 format: direct text access
            text = completion.text
        elif hasattr(completion, "message") and hasattr(completion.message, "content"):
            # V2 format: nested structure (message.content[].text)
            # V2 responses may have multiple content items (thinking, text, etc.)
            content_items = completion.message.content
            if content_items and len(content_items) > 0:
                # Find the text content item (skip thinking/other types)
                text = None
                for item in content_items:
                    if (
                        hasattr(item, "type")
                        and item.type == "text"
                        and hasattr(item, "text")
                    ):
                        text = item.text
                        break

                if text is None:
                    raise ResponseParsingError(
                        "Cohere V2 response has no text content item",
                        mode="COHERE_TOOLS",
                        raw_response=completion,
                    )
            else:
                raise ResponseParsingError(
                    "Cohere V2 response has no content",
                    mode="COHERE_TOOLS",
                    raw_response=completion,
                )
        else:
            raise ResponseParsingError(
                f"Unsupported Cohere response format. Expected tool_calls or text content. "
                f"Got: {type(completion)}",
                mode="COHERE_TOOLS",
                raw_response=completion,
            )

        # Extract JSON from text (for prompt-based approach)
        extra_text = extract_json_from_codeblock(text)
        return cls.model_validate_json(
            extra_text, context=validation_context, strict=strict
        )

    @classmethod
    def parse_writer_tools(
        cls: type[BaseModel],
        completion: ChatCompletion,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> BaseModel:
        message = completion.choices[0].message
        tool_calls = message.tool_calls if message.tool_calls else "{}"
        assert len(tool_calls) == 1, (
            "Instructor does not support multiple tool calls, use List[Model] instead"
        )
        assert tool_calls[0].function.name == cls.openai_schema["name"], (
            "Tool name does not match"
        )
        loaded_args = json.loads(tool_calls[0].function.arguments)
        return cls.model_validate_json(
            json.dumps(loaded_args) if isinstance(loaded_args, dict) else loaded_args,
            context=validation_context,
            strict=strict,
        )

    @classmethod
    def parse_writer_json(
        cls: type[BaseModel],
        completion: ChatCompletion,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> BaseModel:
        _handle_incomplete_output(completion)

        message = completion.choices[0].message.content or ""
        json_content = extract_json_from_codeblock(message)

        if strict:
            return cls.model_validate_json(
                json_content, context=validation_context, strict=True
            )
        else:
            parsed = json.loads(json_content, strict=False)
            return cls.model_validate(parsed, context=validation_context, strict=False)

    @classmethod
    def parse_functions(
        cls: type[BaseModel],
        completion: ChatCompletion,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> BaseModel:
        message = completion.choices[0].message
        assert (
            message.function_call.name == cls.openai_schema["name"]  # type: ignore[index]
        ), "Function name does not match"
        return cls.model_validate_json(
            message.function_call.arguments,  # type: ignore[attr-defined]
            context=validation_context,
            strict=strict,
        )

    @classmethod
    def parse_responses_tools(
        cls: type[BaseModel],
        completion: Any,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> BaseModel:
        from openai.types.responses import ResponseFunctionToolCall

        tool_call_message = None
        for message in completion.output:
            if isinstance(message, ResponseFunctionToolCall):
                if message.name == cls.openai_schema["name"]:
                    tool_call_message = message
                    break
        if not tool_call_message:
            raise ResponseParsingError(
                f"Required tool call '{cls.openai_schema['name']}' not found in response",
                mode="RESPONSES_TOOLS",
                raw_response=completion,
            )

        return cls.model_validate_json(
            tool_call_message.arguments,  # type: ignore[attr-defined]
            context=validation_context,
            strict=strict,
        )

    @classmethod
    def parse_tools(
        cls: type[BaseModel],
        completion: ChatCompletion,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> BaseModel:
        message = completion.choices[0].message
        # this field seems to be missing when using instructor with some other tools (e.g. litellm)
        # trying to fix this by adding a check

        if hasattr(message, "refusal"):
            assert message.refusal is None, (
                f"Unable to generate a response due to {message.refusal}"
            )
        assert len(message.tool_calls or []) == 1, (
            f"Instructor does not support multiple tool calls, use List[Model] instead"
        )
        tool_call = message.tool_calls[0]  # type: ignore
        assert (
            tool_call.function.name == cls.openai_schema["name"]  # type: ignore[index]
        ), "Tool name does not match"
        return cls.model_validate_json(
            tool_call.function.arguments,  # type: ignore
            context=validation_context,
            strict=strict,
        )

    @classmethod
    def parse_mistral_structured_outputs(
        cls: type[BaseModel],
        completion: ChatCompletion,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> BaseModel:
        if not completion.choices or len(completion.choices) > 1:
            raise ConfigurationError(
                "Instructor does not support multiple tool calls in MISTRAL_STRUCTURED_OUTPUTS mode. "
                "Use list[Model] instead to handle multiple items."
            )

        message = completion.choices[0].message

        return cls.model_validate_json(
            message.content, context=validation_context, strict=strict
        )

    @classmethod
    def parse_json(
        cls: type[BaseModel],
        completion: ChatCompletion,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> BaseModel:
        """Parse JSON mode responses using the optimized extraction and validation."""
        # Check for incomplete output
        _handle_incomplete_output(completion)

        # Extract text from the response
        message = _extract_text_content(completion)
        if not message:
            # Fallback for OpenAI format if _extract_text_content doesn't handle it
            message = completion.choices[0].message.content or ""

        # Extract JSON from the text
        json_content = extract_json_from_codeblock(message)

        # Validate the model from the JSON
        return _validate_model_from_json(cls, json_content, validation_context, strict)


def openai_schema(cls: type[BaseModel]) -> OpenAISchema:
    """
    Wrap a Pydantic model class to add OpenAISchema functionality.
    """
    if not issubclass(cls, BaseModel):
        raise ConfigurationError(
            f"response_model must be a Pydantic BaseModel subclass, got {type(cls).__name__}"
        )

    # Create the wrapped model
    schema = wraps(cls, updated=())(
        create_model(
            cls.__name__ if hasattr(cls, "__name__") else str(cls),
            __base__=(cls, OpenAISchema),
        )
    )

    return cast(OpenAISchema, schema)
