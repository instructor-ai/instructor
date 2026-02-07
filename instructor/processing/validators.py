"""Validators that extend OpenAISchema for structured outputs.

This module provides validation classes for LLM-based validation:
- Validator: Original simple validator (backward compatible)
- ValidationResult[T]: Enhanced generic validator with confidence scoring
"""

from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .function_calls import OpenAISchema


# Type variable for generic validation result
T = TypeVar("T")


class Validator(OpenAISchema):
    """
    Validate if an attribute is correct and if not,
    return a new value with an error message.
    
    This is the original validator class, maintained for backward compatibility.
    For new implementations, consider using ValidationResult[T] for better type safety.
    """

    is_valid: bool = Field(
        default=True,
        description="Whether the attribute is valid based on the requirements",
    )
    reason: Optional[str] = Field(
        default=None,
        description="The error message if the attribute is not valid, otherwise None",
    )
    fixed_value: Optional[str] = Field(
        default=None,
        description="If the attribute is not valid, suggest a new value for the attribute",
    )


class ValidationResult(BaseModel, Generic[T]):
    """
    Enhanced validator supporting any type and confidence scoring.
    
    This is a Pydantic v2-native validator that provides:
    - Generic type support for fixed_value (not just strings)
    - Confidence scoring for LLM-based validation decisions
    - Multiple error message support
    - Strict model configuration
    
    Example usage:
        ```python
        from instructor.processing.validators import ValidationResult
        from pydantic import BaseModel
        
        class UserName(BaseModel):
            first: str
            last: str
        
        # The LLM can return ValidationResult with properly typed fixed_value
        result: ValidationResult[UserName] = client.chat.completions.create(
            response_model=ValidationResult[UserName],
            messages=[...],
        )
        
        if not result.is_valid:
            print(f"Confidence: {result.confidence}")
            print(f"Errors: {result.errors}")
            if result.fixed_value:
                corrected_name = result.fixed_value  # Type: UserName
        ```
    
    Attributes:
        is_valid: Whether the validated value passes all requirements.
        confidence: Confidence score (0.0-1.0) for LLM-based validation decisions.
            Higher values indicate more certainty in the validation result.
        errors: List of validation error messages. Can contain multiple specific issues.
        reason: Primary error message (for backward compatibility with Validator).
        fixed_value: Suggested corrected value of the same type as the input.
    """
    
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )
    
    is_valid: bool = Field(
        default=True,
        description="Whether the attribute is valid based on the requirements",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Confidence score (0.0-1.0) for the validation result. "
            "1.0 means completely certain, 0.0 means no confidence."
        ),
    )
    errors: list[str] = Field(
        default_factory=list,
        description="List of validation error messages if the attribute is not valid",
    )
    reason: Optional[str] = Field(
        default=None,
        description=(
            "Primary error message if the attribute is not valid. "
            "For backward compatibility with Validator class."
        ),
    )
    fixed_value: Optional[T] = Field(
        default=None,
        description=(
            "Suggested corrected value of the same type as the validated input. "
            "Only provided when is_valid is False and a fix is possible."
        ),
    )
    
    @field_validator("errors", mode="before")
    @classmethod
    def ensure_errors_list(cls, v: Any) -> list[str]:
        """Ensure errors is always a list."""
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return list(v)
    
    def model_post_init(self, __context: Any) -> None:
        """Sync reason with errors for consistency."""
        # If reason is provided but errors is empty, populate errors
        if self.reason and not self.errors:
            self.errors = [self.reason]
        # If errors exist but reason is empty, set reason to first error
        elif self.errors and not self.reason:
            object.__setattr__(self, "reason", self.errors[0])
    
    @property
    def is_confident(self) -> bool:
        """Check if validation result has high confidence (>= 0.8)."""
        return self.confidence >= 0.8
    
    def get_summary(self) -> str:
        """Get a human-readable summary of the validation result."""
        if self.is_valid:
            return f"Valid (confidence: {self.confidence:.0%})"
        error_summary = "; ".join(self.errors) if self.errors else "Unknown error"
        return f"Invalid (confidence: {self.confidence:.0%}): {error_summary}"

