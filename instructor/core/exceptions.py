from __future__ import annotations

from textwrap import dedent
from typing import Any, NamedTuple
from jinja2 import Template

from .errors import format_mode_error, format_config_error, format_provider_error


class InstructorError(Exception):
    """Base exception for all Instructor-specific errors."""

    failed_attempts: list[FailedAttempt] | None = None

    @classmethod
    def from_exception(
        cls, exception: Exception, failed_attempts: list[FailedAttempt] | None = None
    ):
        return cls(str(exception), failed_attempts=failed_attempts)

    def __init__(
        self,
        *args: Any,
        failed_attempts: list[FailedAttempt] | None = None,
        **kwargs: dict[str, Any],
    ):
        self.failed_attempts = failed_attempts
        super().__init__(*args, **kwargs)

    def __str__(self) -> str:
        # If no failed attempts, use the standard exception string representation
        if not self.failed_attempts:
            return super().__str__()

        template = Template(
            dedent(
                """
                <failed_attempts>
                {% for attempt in failed_attempts %}
                <generation number="{{ attempt.attempt_number }}">
                <exception>
                    {{ attempt.exception }}
                </exception>
                <completion>
                    {{ attempt.completion }}
                </completion>
                </generation>
                {% endfor %}
                </failed_attempts>

                <last_exception>
                    {{ last_exception }}
                </last_exception>
                """
            ).strip()
        )
        return template.render(
            last_exception=super().__str__(), failed_attempts=self.failed_attempts
        )


class FailedAttempt(NamedTuple):
    """Represents a single failed retry attempt."""

    attempt_number: int
    exception: Exception
    completion: Any | None = None


class IncompleteOutputException(InstructorError):
    """Exception raised when the output from LLM is incomplete due to max tokens limit reached."""

    def __init__(
        self,
        *args: Any,
        last_completion: Any | None = None,
        message: str = "The output is incomplete due to a max_tokens length limit.",
        **kwargs: dict[str, Any],
    ):
        self.last_completion = last_completion
        super().__init__(message, *args, **kwargs)


class InstructorRetryException(InstructorError):
    """Exception raised when all retry attempts have been exhausted."""

    def __init__(
        self,
        *args: Any,
        last_completion: Any | None = None,
        messages: list[Any] | None = None,
        n_attempts: int,
        total_usage: int,
        create_kwargs: dict[str, Any] | None = None,
        failed_attempts: list[FailedAttempt] | None = None,
        **kwargs: dict[str, Any],
    ):
        self.last_completion = last_completion
        self.messages = messages
        self.n_attempts = n_attempts
        self.total_usage = total_usage
        self.create_kwargs = create_kwargs
        super().__init__(*args, failed_attempts=failed_attempts, **kwargs)


class ValidationError(InstructorError):
    """Exception raised when response validation fails."""

    pass


class ProviderError(InstructorError):
    """Exception raised for provider-specific errors."""

    def __init__(
        self,
        provider: str,
        message: str,
        context: dict[str, Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ):
        self.provider = provider
        self.context = context
        formatted_message = format_provider_error(provider, message, context)
        super().__init__(formatted_message, *args, **kwargs)


class ConfigurationError(InstructorError):
    """Exception raised for configuration-related errors."""

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        env_var: str | None = None,
        missing_key: str | None = None,
        *args: Any,
        **kwargs: Any,
    ):
        self.provider = provider
        self.env_var = env_var
        self.missing_key = missing_key
        
        # If provider is specified, use rich formatting
        if provider:
            formatted_message = format_config_error(provider, env_var, missing_key)
            # Prepend the original message if provided
            if message and message != formatted_message:
                formatted_message = f"{message}\n\n{formatted_message}"
        else:
            formatted_message = message
            
        super().__init__(formatted_message, *args, **kwargs)


class ModeError(InstructorError):
    """Exception raised when an invalid mode is used for a provider."""

    def __init__(
        self,
        mode: str,
        provider: str,
        valid_modes: list[str],
        *args: Any,
        **kwargs: Any,
    ):
        self.mode = mode
        self.provider = provider
        self.valid_modes = valid_modes
        message = format_mode_error(mode, provider, valid_modes)
        super().__init__(message, *args, **kwargs)


class ClientError(InstructorError):
    """Exception raised for client initialization or usage errors."""

    pass


class AsyncValidationError(ValueError, InstructorError):
    """Exception raised during async validation."""

    errors: list[ValueError]
