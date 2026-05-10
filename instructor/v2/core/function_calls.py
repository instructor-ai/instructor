# type: ignore
import inspect
import json
import logging
import warnings
import re
from functools import wraps
from typing import Any, Optional, TypeVar, cast
from openai.types.chat import ChatCompletion
from pydantic import (
    BaseModel,
    ConfigDict,
    TypeAdapter,
    create_model,
)

from instructor.v2.core.errors import (
    IncompleteOutputException,
    ResponseParsingError,
    ConfigurationError,
)
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider, normalize_mode_for_provider, provider_from_mode
from instructor.v2.core.utils import classproperty
from instructor.v2.core.json import extract_json_from_codeblock
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


class ResponseSchema(BaseModel):
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
        provider: Provider = Provider.OPENAI,
    ) -> BaseModel:
        """Execute the function from the response of an openai chat completion

        Parameters:
            completion (openai.ChatCompletion): The response from an openai chat completion
            strict (bool): Whether to use strict json parsing
            mode (Mode): The completion mode
            provider (Provider): The provider for handler lookup

        Returns:
            cls (ResponseSchema): An instance of the class
        """

        import importlib

        from instructor.v2.core.registry import mode_registry

        importlib.import_module("instructor.v2")

        provider = provider_from_mode(mode, provider)
        mode = normalize_mode_for_provider(mode, provider)
        handlers = mode_registry.get_handlers(provider, mode)
        return handlers.response_parser(
            response=completion,
            response_model=cls,
            validation_context=validation_context,
            strict=strict,
            stream=False,
            is_async=False,
        )

    @classmethod
    def _parse_with_registry(
        cls: type[BaseModel],
        completion: Any,
        *,
        mode: Mode,
        provider: Provider,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
        warning: Optional[str] = None,
    ) -> BaseModel:
        if warning:
            warnings.warn(warning, DeprecationWarning, stacklevel=2)
        return cls.from_response(
            completion,
            validation_context=validation_context,
            strict=strict,
            mode=mode,
            provider=provider,
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
        """Legacy Anthropic tools parser (deprecated)."""
        return cls._parse_with_registry(
            completion,
            mode=Mode.ANTHROPIC_TOOLS,
            provider=Provider.ANTHROPIC,
            validation_context=validation_context,
            strict=strict,
            warning=(
                "ResponseSchema.parse_anthropic_tools is deprecated. "
                "Use process_response(..., provider=Provider.ANTHROPIC, mode=Mode.TOOLS) "
                "or ResponseSchema.from_response with core modes."
            ),
        )

    @classmethod
    def parse_anthropic_json(
        cls: type[BaseModel],
        completion: ChatCompletion,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> BaseModel:
        """Legacy Anthropic JSON parser (deprecated)."""
        return cls._parse_with_registry(
            completion,
            mode=Mode.ANTHROPIC_JSON,
            provider=Provider.ANTHROPIC,
            validation_context=validation_context,
            strict=strict,
            warning=(
                "ResponseSchema.parse_anthropic_json is deprecated. "
                "Use process_response(..., provider=Provider.ANTHROPIC, mode=Mode.JSON) "
                "or ResponseSchema.from_response with core modes."
            ),
        )

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
    def parse_gemini_tools(
        cls: type[BaseModel],
        completion: Any,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> BaseModel:
        try:
            function_call = completion.candidates[0].content.parts[0].function_call
        except Exception as exc:
            raise ResponseParsingError(
                "No tool call found in Gemini response",
                mode="GEMINI_TOOLS",
                raw_response=completion,
            ) from exc

        args = getattr(function_call, "args", None)
        if args is None and hasattr(type(function_call), "to_dict"):
            try:
                resp_dict = type(function_call).to_dict(function_call)
            except Exception:
                resp_dict = {}
            args = resp_dict.get("args")

        if args is None:
            raise ResponseParsingError(
                "No tool call args found in Gemini response",
                mode="GEMINI_TOOLS",
                raw_response=completion,
            )

        return cls.model_validate(args, context=validation_context, strict=strict)

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
        """Legacy OpenAI FUNCTIONS parser (deprecated)."""
        return cls._parse_with_registry(
            completion,
            mode=Mode.FUNCTIONS,
            provider=Provider.OPENAI,
            validation_context=validation_context,
            strict=strict,
            warning=(
                "ResponseSchema.parse_functions is deprecated. "
                "Use process_response(..., mode=Mode.TOOLS) or ResponseSchema.from_response."
            ),
        )

    @classmethod
    def parse_responses_tools(
        cls: type[BaseModel],
        completion: Any,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> BaseModel:
        """Legacy OpenAI Responses Tools parser (deprecated)."""
        return cls._parse_with_registry(
            completion,
            mode=Mode.RESPONSES_TOOLS,
            provider=Provider.OPENAI,
            validation_context=validation_context,
            strict=strict,
            warning=(
                "ResponseSchema.parse_responses_tools is deprecated. "
                "Use process_response(..., mode=Mode.RESPONSES_TOOLS) or ResponseSchema.from_response."
            ),
        )

    @classmethod
    def parse_tools(
        cls: type[BaseModel],
        completion: ChatCompletion,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> BaseModel:
        """Legacy OpenAI tools parser (deprecated)."""
        return cls._parse_with_registry(
            completion,
            mode=Mode.TOOLS,
            provider=Provider.OPENAI,
            validation_context=validation_context,
            strict=strict,
            warning=(
                "ResponseSchema.parse_tools is deprecated. "
                "Use process_response(..., mode=Mode.TOOLS) or ResponseSchema.from_response."
            ),
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
        """Legacy JSON parser (deprecated)."""
        return cls._parse_with_registry(
            completion,
            mode=Mode.JSON,
            provider=Provider.OPENAI,
            validation_context=validation_context,
            strict=strict,
            warning=(
                "ResponseSchema.parse_json is deprecated. "
                "Use process_response(..., mode=Mode.JSON) or ResponseSchema.from_response."
            ),
        )


def response_schema(cls: type[BaseModel]) -> ResponseSchema:
    """Wrap a Pydantic model class to add ResponseSchema behavior."""
    if not inspect.isclass(cls) or not issubclass(cls, BaseModel):
        got = cls.__name__ if inspect.isclass(cls) else type(cls).__name__
        raise TypeError(
            f"response_model must be a subclass of pydantic.BaseModel, got {got}"
        )

    # Create the wrapped model
    schema = wraps(cls, updated=())(
        create_model(
            cls.__name__ if hasattr(cls, "__name__") else str(cls),
            __base__=(cls, ResponseSchema),
        )
    )

    return cast(ResponseSchema, schema)


# Backward compatibility aliases
openai_schema = response_schema
OpenAISchema = ResponseSchema
