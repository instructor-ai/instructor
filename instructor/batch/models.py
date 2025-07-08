"""
Data models and types for batch processing.

This module contains all the Pydantic models, enums, and type definitions
used throughout the batch processing system.
"""

from __future__ import annotations
from typing import Any, Union, TypeVar, Optional, List, Dict, Generic
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from enum import Enum

T = TypeVar("T", bound=BaseModel)


class BatchSuccess(BaseModel, Generic[T]):
    """Successful batch result with custom_id"""

    custom_id: str
    result: T
    success: bool = True

    class Config:
        arbitrary_types_allowed = True


class BatchError(BaseModel):
    """Error information for failed batch requests"""

    custom_id: str
    error_type: str
    error_message: str
    success: bool = False
    raw_data: Optional[Dict[str, Any]] = None


class BatchStatus(str, Enum):
    """Normalized batch status across providers"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class BatchTimestamps(BaseModel):
    """Comprehensive timestamp tracking"""

    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None  # in_progress_at, processing start
    completed_at: Optional[datetime] = None  # completed_at, ended_at
    failed_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    expired_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None


class BatchRequestCounts(BaseModel):
    """Unified request counts across providers"""

    total: Optional[int] = None

    # OpenAI fields
    completed: Optional[int] = None
    failed: Optional[int] = None

    # Anthropic fields
    processing: Optional[int] = None
    succeeded: Optional[int] = None
    errored: Optional[int] = None
    cancelled: Optional[int] = None
    expired: Optional[int] = None


class BatchErrorInfo(BaseModel):
    """Batch-level error information"""

    error_type: Optional[str] = None
    error_message: Optional[str] = None
    error_code: Optional[str] = None


class BatchFiles(BaseModel):
    """File references for batch job"""

    input_file_id: Optional[str] = None
    output_file_id: Optional[str] = None
    error_file_id: Optional[str] = None
    results_url: Optional[str] = None  # Anthropic


class BatchJobInfo(BaseModel):
    """Enhanced unified batch job information with comprehensive provider support"""

    # Core identifiers
    id: str
    provider: str

    # Status information
    status: BatchStatus
    raw_status: str  # Original provider status

    # Timing information
    timestamps: BatchTimestamps

    # Request tracking
    request_counts: BatchRequestCounts

    # File references
    files: BatchFiles

    # Error information
    error: Optional[BatchErrorInfo] = None

    # Provider-specific data
    metadata: Dict[str, Any] = Field(default_factory=dict)
    raw_data: Optional[Dict[str, Any]] = None

    # Additional fields
    model: Optional[str] = None
    endpoint: Optional[str] = None
    completion_window: Optional[str] = None

    @classmethod
    def from_openai(cls, batch_data: Dict[str, Any]) -> BatchJobInfo:
        """Create from OpenAI batch response"""
        # Normalize status
        status_map = {
            "validating": BatchStatus.PENDING,
            "in_progress": BatchStatus.PROCESSING,
            "finalizing": BatchStatus.PROCESSING,
            "completed": BatchStatus.COMPLETED,
            "failed": BatchStatus.FAILED,
            "expired": BatchStatus.EXPIRED,
            "cancelled": BatchStatus.CANCELLED,
            "cancelling": BatchStatus.CANCELLED,
        }

        # Parse timestamps
        timestamps = BatchTimestamps(
            created_at=datetime.fromtimestamp(batch_data["created_at"], tz=timezone.utc)
            if batch_data.get("created_at")
            else None,
            started_at=datetime.fromtimestamp(
                batch_data["in_progress_at"], tz=timezone.utc
            )
            if batch_data.get("in_progress_at")
            else None,
            completed_at=datetime.fromtimestamp(
                batch_data["completed_at"], tz=timezone.utc
            )
            if batch_data.get("completed_at")
            else None,
            failed_at=datetime.fromtimestamp(batch_data["failed_at"], tz=timezone.utc)
            if batch_data.get("failed_at")
            else None,
            cancelled_at=datetime.fromtimestamp(
                batch_data["cancelled_at"], tz=timezone.utc
            )
            if batch_data.get("cancelled_at")
            else None,
            expired_at=datetime.fromtimestamp(batch_data["expired_at"], tz=timezone.utc)
            if batch_data.get("expired_at")
            else None,
            expires_at=datetime.fromtimestamp(batch_data["expires_at"], tz=timezone.utc)
            if batch_data.get("expires_at")
            else None,
        )

        # Parse request counts
        request_counts_data = batch_data.get("request_counts", {})
        request_counts = BatchRequestCounts(
            total=request_counts_data.get("total"),
            completed=request_counts_data.get("completed"),
            failed=request_counts_data.get("failed"),
        )

        # Parse files
        files = BatchFiles(
            input_file_id=batch_data.get("input_file_id"),
            output_file_id=batch_data.get("output_file_id"),
            error_file_id=batch_data.get("error_file_id"),
        )

        # Parse error information
        error = None
        if batch_data.get("errors"):
            error_data = batch_data["errors"]
            error = BatchErrorInfo(
                error_type=error_data.get("type"),
                error_message=error_data.get("message"),
                error_code=error_data.get("code"),
            )

        return cls(
            id=batch_data["id"],
            provider="openai",
            status=status_map.get(batch_data["status"], BatchStatus.PENDING),
            raw_status=batch_data["status"],
            timestamps=timestamps,
            request_counts=request_counts,
            files=files,
            error=error,
            metadata=batch_data.get("metadata", {}),
            raw_data=batch_data,
            endpoint=batch_data.get("endpoint"),
            completion_window=batch_data.get("completion_window"),
        )

    @classmethod
    def from_anthropic(cls, batch_data: Dict[str, Any]) -> BatchJobInfo:
        """Create from Anthropic batch response"""
        # Normalize status
        status_map = {
            "in_progress": BatchStatus.PROCESSING,
            "ended": BatchStatus.COMPLETED,
            "failed": BatchStatus.FAILED,
            "cancelled": BatchStatus.CANCELLED,
            "expired": BatchStatus.EXPIRED,
        }

        # Parse timestamps
        def parse_iso_timestamp(timestamp_value):
            if not timestamp_value:
                return None
            try:
                # Handle different timestamp format variations
                if isinstance(timestamp_value, datetime):
                    return timestamp_value
                elif isinstance(timestamp_value, str):
                    return datetime.fromisoformat(
                        timestamp_value.replace("Z", "+00:00")
                    )
                else:
                    return None
            except (ValueError, AttributeError):
                return None

        timestamps = BatchTimestamps(
            created_at=parse_iso_timestamp(batch_data.get("created_at")),
            started_at=parse_iso_timestamp(
                batch_data.get("created_at")
            ),  # Anthropic doesn't provide started_at, use created_at
            cancelled_at=parse_iso_timestamp(batch_data.get("cancel_initiated_at")),
            completed_at=parse_iso_timestamp(batch_data.get("ended_at")),
            expires_at=parse_iso_timestamp(batch_data.get("expires_at")),
        )

        # Parse request counts
        request_counts_data = batch_data.get("request_counts", {})
        request_counts = BatchRequestCounts(
            processing=request_counts_data.get("processing"),
            succeeded=request_counts_data.get("succeeded"),
            errored=request_counts_data.get("errored"),
            cancelled=request_counts_data.get(
                "canceled"
            ),  # Note: Anthropic uses "canceled"
            expired=request_counts_data.get("expired"),
            total=request_counts_data.get("processing", 0)
            + request_counts_data.get("succeeded", 0)
            + request_counts_data.get("errored", 0),
        )

        # Parse files
        files = BatchFiles(
            results_url=batch_data.get("results_url"),
        )

        return cls(
            id=batch_data["id"],
            provider="anthropic",
            status=status_map.get(batch_data["processing_status"], BatchStatus.PENDING),
            raw_status=batch_data["processing_status"],
            timestamps=timestamps,
            request_counts=request_counts,
            files=files,
            raw_data=batch_data,
        )


# Union type for batch results - like a Maybe/Result type
BatchResult = Union[BatchSuccess[T], BatchError]
