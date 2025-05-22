"""Enhanced exceptions with structured error messages and debugging context.

This module provides a comprehensive exception hierarchy with:
- Structured error messages with actionable guidance
- Request/response context for debugging
- Provider-specific error handling
- Automatic error categorization
"""

from typing import Any, Optional
import json
from datetime import datetime
import traceback


class InstructorError(Exception):
    """Base exception for all Instructor errors.

    All Instructor exceptions inherit from this class to allow
    catching any Instructor-specific error.
    """

    def __init__(
        self,
        message: str,
        *,
        error_code: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
        guidance: Optional[str] = None,
        request_context: Optional[dict[str, Any]] = None,
        response_context: Optional[dict[str, Any]] = None,
    ):
        """Initialize InstructorError with enhanced context.

        Args:
            message: The error message
            error_code: Optional error code for categorization
            details: Additional error details as dict
            guidance: Actionable guidance for resolving the error
            request_context: Request details for debugging
            response_context: Response details for debugging
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.guidance = guidance
        self.request_context = request_context
        self.response_context = response_context
        self.timestamp = datetime.utcnow().isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for structured logging."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
            "guidance": self.guidance,
            "request_context": self.request_context,
            "response_context": self.response_context,
            "timestamp": self.timestamp,
            "traceback": traceback.format_exc(),
        }

    def __str__(self) -> str:
        """Enhanced string representation with guidance."""
        parts = [f"{self.__class__.__name__}: {self.message}"]

        if self.error_code:
            parts.append(f"Error Code: {self.error_code}")

        if self.details:
            parts.append(f"Details: {json.dumps(self.details, indent=2)}")

        if self.guidance:
            parts.append(f"\nGuidance: {self.guidance}")

        if self.request_context:
            parts.append(
                f"\nRequest Context: {json.dumps(self.request_context, indent=2)}"
            )

        return "\n".join(parts)


class ProviderError(InstructorError):
    """Base exception for provider-specific errors."""

    def __init__(self, message: str, provider: str, **kwargs):
        super().__init__(message, **kwargs)
        self.provider = provider
        self.details["provider"] = provider


class OpenAIError(ProviderError):
    """OpenAI-specific errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, provider="openai", **kwargs)


class AnthropicError(ProviderError):
    """Anthropic-specific errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, provider="anthropic", **kwargs)


class ValidationError(InstructorError):
    """Validation errors from Pydantic or custom validators."""

    def __init__(
        self,
        message: str,
        *,
        validation_errors: Optional[list[dict[str, Any]]] = None,
        model_name: Optional[str] = None,
        **kwargs,
    ):
        # Provide helpful guidance based on common validation errors
        guidance = self._generate_validation_guidance(validation_errors)

        super().__init__(message, guidance=guidance, **kwargs)
        self.validation_errors = validation_errors or []
        self.model_name = model_name

        if validation_errors:
            self.details["validation_errors"] = validation_errors
        if model_name:
            self.details["model_name"] = model_name

    def _generate_validation_guidance(
        self, errors: Optional[list[dict[str, Any]]]
    ) -> str:
        """Generate helpful guidance based on validation errors."""
        if not errors:
            return "Check your Pydantic model definition and ensure the LLM response matches the expected schema."

        guidance_parts = []

        for error in errors:
            error_type = error.get("type", "")
            loc = error.get("loc", [])

            if error_type == "missing":
                field = ".".join(str(l) for l in loc)
                guidance_parts.append(
                    f"- Field '{field}' is required. Ensure your prompt asks for this field explicitly."
                )
            elif error_type == "type_error":
                field = ".".join(str(l) for l in loc)
                guidance_parts.append(
                    f"- Field '{field}' has wrong type. Check the expected type in your model."
                )
            elif error_type == "value_error":
                guidance_parts.append(
                    f"- Validation failed for custom validator. Review your validation logic."
                )

        if guidance_parts:
            return "Validation issues found:\n" + "\n".join(guidance_parts)

        return "Review your model schema and validation rules."


class ConfigurationError(InstructorError):
    """Configuration-related errors."""

    def __init__(self, message: str, *, config_key: Optional[str] = None, **kwargs):
        guidance = self._generate_config_guidance(config_key, message)
        super().__init__(message, guidance=guidance, **kwargs)
        self.config_key = config_key
        if config_key:
            self.details["config_key"] = config_key

    def _generate_config_guidance(self, config_key: Optional[str], message: str) -> str:
        """Generate configuration-specific guidance."""
        common_configs = {
            "api_key": "Set your API key via environment variable or pass it to the client constructor.",
            "base_url": "Ensure the base URL is correct and the service is accessible.",
            "model": "Check that the model name is valid for this provider.",
            "max_retries": "Set max_retries to a positive integer (recommended: 3-5).",
        }

        if config_key and config_key in common_configs:
            return common_configs[config_key]

        if "api" in message.lower() and "key" in message.lower():
            return common_configs["api_key"]

        return "Check your configuration parameters and provider documentation."


class ParsingError(InstructorError):
    """Errors during response parsing."""

    def __init__(
        self,
        message: str,
        *,
        raw_response: Optional[str] = None,
        expected_format: Optional[str] = None,
        **kwargs,
    ):
        guidance = self._generate_parsing_guidance(raw_response, expected_format)
        super().__init__(message, guidance=guidance, **kwargs)
        self.raw_response = raw_response
        self.expected_format = expected_format

        if raw_response:
            self.details["raw_response"] = raw_response[:500]  # Truncate for logging
        if expected_format:
            self.details["expected_format"] = expected_format

    def _generate_parsing_guidance(
        self, raw_response: Optional[str], expected_format: Optional[str]
    ) -> str:
        """Generate parsing-specific guidance."""
        guidance_parts = []

        if raw_response and not raw_response.strip():
            guidance_parts.append(
                "- Response is empty. Check if the model completed successfully."
            )
        elif raw_response and not (
            raw_response.startswith("{") or raw_response.startswith("[")
        ):
            guidance_parts.append(
                "- Response doesn't appear to be JSON. Ensure you're using JSON mode."
            )

        if expected_format:
            guidance_parts.append(f"- Expected format: {expected_format}")

        guidance_parts.extend(
            [
                "- Try using a more specific prompt that requests JSON output",
                "- Consider using Mode.TOOLS instead of Mode.JSON for better reliability",
                "- Ensure the model supports structured outputs",
            ]
        )

        return "\n".join(guidance_parts)


class TokenLimitError(InstructorError):
    """Token limit exceeded errors."""

    def __init__(
        self,
        message: str,
        *,
        token_count: Optional[int] = None,
        token_limit: Optional[int] = None,
        **kwargs,
    ):
        guidance = self._generate_token_guidance(token_count, token_limit)
        super().__init__(message, guidance=guidance, error_code="TOKEN_LIMIT", **kwargs)

        if token_count:
            self.details["token_count"] = token_count
        if token_limit:
            self.details["token_limit"] = token_limit

    def _generate_token_guidance(
        self, token_count: Optional[int], token_limit: Optional[int]
    ) -> str:
        """Generate token limit guidance."""
        guidance_parts = [
            "Token limit exceeded. Try:",
            "- Reducing the prompt length",
            "- Using a smaller response model",
            "- Breaking the task into smaller chunks",
            "- Using a model with higher token limits",
        ]

        if token_count and token_limit:
            guidance_parts.insert(
                1, f"- Current: {token_count} tokens, Limit: {token_limit} tokens"
            )

        return "\n".join(guidance_parts)


class RetryError(InstructorError):
    """Errors after retry exhaustion."""

    def __init__(
        self,
        message: str,
        *,
        attempts: int,
        last_error: Optional[Exception] = None,
        **kwargs,
    ):
        guidance = (
            f"Failed after {attempts} attempts. Consider:\n"
            "- Increasing max_retries\n"
            "- Improving your prompt clarity\n"
            "- Checking if the model can handle your request\n"
            "- Using a more capable model"
        )

        super().__init__(
            message, guidance=guidance, error_code="RETRY_EXHAUSTED", **kwargs
        )
        self.attempts = attempts
        self.last_error = last_error
        self.details["attempts"] = attempts
        if last_error:
            self.details["last_error"] = str(last_error)


# Legacy exceptions for backward compatibility
class IncompleteOutputException(ValidationError):
    """Raised when the response is incomplete."""

    def __init__(self, **kwargs):
        super().__init__(
            "Incomplete output from model",
            guidance="The model's response was cut off. Try reducing the response size or using a model with higher token limits.",
            error_code="INCOMPLETE_OUTPUT",
            **kwargs,
        )


class InstructorRetryException(RetryError):
    """Legacy retry exception for backward compatibility."""

    def __init__(
        self,
        *,
        n_attempts: int,
        last_error: Optional[Exception] = None,
        total_usage: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(
            f"Failed after {n_attempts} retries",
            attempts=n_attempts,
            last_error=last_error,
            **kwargs,
        )
        self.n_attempts = n_attempts
        self.total_usage = total_usage
        if total_usage:
            self.details["total_usage"] = total_usage


def create_error_from_response(
    provider: str, response: Any, message: Optional[str] = None, **kwargs
) -> InstructorError:
    """Factory function to create appropriate error from provider response.

    Args:
        provider: Name of the provider
        response: The error response from the provider
        message: Optional custom message
        **kwargs: Additional error context

    Returns:
        Appropriate InstructorError subclass
    """
    error_mapping = {
        "openai": OpenAIError,
        "anthropic": AnthropicError,
    }

    error_class = error_mapping.get(provider, ProviderError)

    # Extract relevant info from response
    if hasattr(response, "error"):
        error_info = response.error
        default_message = getattr(error_info, "message", str(error_info))
        error_code = getattr(error_info, "code", None)
        error_type = getattr(error_info, "type", None)

        if error_code:
            kwargs["error_code"] = error_code
        if error_type:
            kwargs.setdefault("details", {})["error_type"] = error_type
    else:
        default_message = str(response)

    final_message = message or default_message

    # Check for specific error patterns
    if "token" in final_message.lower() and "limit" in final_message.lower():
        return TokenLimitError(final_message, **kwargs)
    elif "rate limit" in final_message.lower():
        kwargs["guidance"] = (
            "Rate limit hit. Consider implementing exponential backoff or reducing request frequency."
        )
        kwargs["error_code"] = "RATE_LIMIT"

    if error_class == ProviderError:
        return error_class(final_message, provider=provider, **kwargs)
    else:
        return error_class(final_message, **kwargs)
