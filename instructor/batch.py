"""
Unified Batch Processing API for Multiple Providers

This module provides a unified interface for batch processing across OpenAI and Anthropic
providers. The API uses a Maybe/Result-like pattern with custom_id
tracking for type-safe handling of batch results.

Supported Providers:
- OpenAI: 50% cost savings on batch requests
- Anthropic: 50% cost savings on batch requests (Message Batches API)

Features:
- Type-safe Maybe/Result pattern for handling successes and errors
- Custom ID tracking for correlating results to original requests
- Unified interface across all providers
- Helper functions for filtering and extracting results

Example usage:
    from instructor.batch import BatchProcessor, filter_successful, extract_results
    from pydantic import BaseModel

    class User(BaseModel):
        name: str
        age: int

    processor = BatchProcessor("openai/gpt-4o-mini", User)
    batch_id = processor.submit_batch("requests.jsonl")

    # Results are BatchSuccess[T] | BatchError union types
    all_results = processor.retrieve_results(batch_id)
    successful_results = filter_successful(all_results)
    extracted_users = extract_results(all_results)

Documentation:
- OpenAI Batch API: https://platform.openai.com/docs/guides/batch
- Anthropic Message Batches: https://docs.anthropic.com/en/api/creating-message-batches
"""

from __future__ import annotations
from typing import Any, Union, TypeVar, Optional, List, Dict, Type, Generic
from collections.abc import Iterable
from pydantic import BaseModel, Field
from instructor.process_response import handle_response_model
import instructor
import uuid
import json
import os
from instructor.auto_client import from_provider
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


def filter_successful(results: List[BatchResult]) -> List[BatchSuccess[T]]:
    """Filter to only successful results"""
    return [r for r in results if r.success]


def filter_errors(results: List[BatchResult]) -> List[BatchError]:
    """Filter to only error results"""
    return [r for r in results if not r.success]


def extract_results(results: List[BatchResult]) -> List[T]:
    """Extract just the result objects from successful results"""
    return [r.result for r in results if r.success]


def get_results_by_custom_id(results: List[BatchResult]) -> Dict[str, BatchResult]:
    """Create a dictionary mapping custom_id to results"""
    return {r.custom_id: r for r in results}


class Function(BaseModel):
    name: str
    description: str
    parameters: Any


class Tool(BaseModel):
    type: str
    function: Function


class RequestBody(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    max_tokens: Optional[int] = Field(default=1000)
    temperature: Optional[float] = Field(default=1.0)
    tools: Optional[list[Tool]]
    tool_choice: Optional[dict[str, Any]]


class BatchModel(BaseModel):
    custom_id: str
    body: RequestBody
    url: str
    method: str


class BatchRequest(BaseModel, Generic[T]):
    """Unified batch request that works across all providers using JSON schema"""

    custom_id: str
    messages: List[Dict[str, Any]]
    response_model: Type[T]
    model: str
    max_tokens: Optional[int] = Field(default=1000)
    temperature: Optional[float] = Field(default=0.1)

    class Config:
        arbitrary_types_allowed = True

    def get_json_schema(self) -> Dict[str, Any]:
        """Generate JSON schema from response_model"""
        return self.response_model.model_json_schema()

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI batch format with JSON schema"""
        schema = self.get_json_schema()

        # OpenAI strict mode requires additionalProperties to be false
        def make_strict_schema(schema_dict):
            """Recursively add additionalProperties: false for OpenAI strict mode"""
            if isinstance(schema_dict, dict):
                if "type" in schema_dict:
                    if schema_dict["type"] == "object":
                        schema_dict["additionalProperties"] = False
                    elif schema_dict["type"] == "array" and "items" in schema_dict:
                        schema_dict["items"] = make_strict_schema(schema_dict["items"])

                # Recursively process properties
                if "properties" in schema_dict:
                    for prop_name, prop_schema in schema_dict["properties"].items():
                        schema_dict["properties"][prop_name] = make_strict_schema(
                            prop_schema
                        )

                # Process definitions/defs
                for key in ["definitions", "$defs"]:
                    if key in schema_dict:
                        for def_name, def_schema in schema_dict[key].items():
                            schema_dict[key][def_name] = make_strict_schema(def_schema)

            return schema_dict

        strict_schema = make_strict_schema(schema.copy())

        return {
            "custom_id": self.custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.model,
                "messages": self.messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": self.response_model.__name__,
                        "strict": True,
                        "schema": strict_schema,
                    },
                },
            },
        }

    def to_anthropic_format(self) -> Dict[str, Any]:
        """Convert to Anthropic batch format with JSON schema"""
        schema = self.get_json_schema()

        # Ensure schema has proper format for Anthropic
        if "type" not in schema:
            schema["type"] = "object"
        if "additionalProperties" not in schema:
            schema["additionalProperties"] = False

        # Extract system message and convert to system parameter
        system_message = None
        filtered_messages = []

        for message in self.messages:
            if message.get("role") == "system":
                system_message = message.get("content", "")
            else:
                filtered_messages.append(message)

        params = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": filtered_messages,
            "tools": [
                {
                    "name": "extract_data",
                    "description": f"Extract data matching the {self.response_model.__name__} schema",
                    "input_schema": schema,
                }
            ],
            "tool_choice": {"type": "tool", "name": "extract_data"},
        }

        # Add system parameter if system message exists
        if system_message:
            params["system"] = system_message

        return {
            "custom_id": self.custom_id,
            "params": params,
        }

    def save_to_file(self, file_path: str, provider: str) -> None:
        """Save batch request to file in provider-specific format"""
        if provider == "openai":
            data = self.to_openai_format()
        elif provider == "anthropic":
            data = self.to_anthropic_format()
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        with open(file_path, "a") as f:
            f.write(json.dumps(data) + "\n")


class BatchProcessor(Generic[T]):
    """Unified batch processor that works across all providers"""

    def __init__(self, model: str, response_model: Type[T]):
        self.model = model
        self.response_model = response_model

        # Parse provider from model string
        try:
            self.provider_name, self.model_name = model.split("/", 1)
        except ValueError:
            raise ValueError(
                'Model string must be in format "provider/model-name" '
                '(e.g. "openai/gpt-4" or "anthropic/claude-3-sonnet")'
            )

        # Get the instructor client
        self.client = from_provider(model)

    def create_batch_from_messages(
        self,
        messages_list: List[List[Dict[str, Any]]],
        file_path: str,
        max_tokens: Optional[int] = 1000,
        temperature: Optional[float] = 0.1,
    ) -> str:
        """Create batch file from list of message conversations

        Args:
            messages_list: List of message conversations, each as a list of message dicts
            file_path: Path to save the batch request file
            max_tokens: Maximum tokens per request
            temperature: Temperature for generation

        Returns:
            The file path where the batch was saved
        """
        # Remove existing file if it exists
        if os.path.exists(file_path):
            os.remove(file_path)

        batch_requests = []
        for i, messages in enumerate(messages_list):
            batch_request = BatchRequest[self.response_model](
                custom_id=f"request-{i}",
                messages=messages,
                response_model=self.response_model,
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            batch_request.save_to_file(file_path, self.provider_name)
            batch_requests.append(batch_request)

        print(f"Created batch file {file_path} with {len(batch_requests)} requests")
        return file_path

    def submit_batch(
        self, file_path: str, metadata: Optional[Dict[str, Any]] = None, **kwargs
    ) -> str:
        """Submit batch job to the provider and return job ID

        Args:
            file_path: Path to the batch request file
            metadata: Optional metadata to attach to the batch job
            **kwargs: Additional provider-specific arguments
        """
        if metadata is None:
            metadata = {"description": "Instructor batch job"}

        if self.provider_name == "openai":
            return self._submit_openai_batch(file_path, metadata=metadata, **kwargs)
        elif self.provider_name == "anthropic":
            return self._submit_anthropic_batch(file_path, metadata=metadata, **kwargs)
        else:
            raise ValueError(
                f"Unsupported provider for batch submission: {self.provider_name}"
            )

    def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """Get batch job status from the provider"""
        if self.provider_name == "openai":
            return self._get_openai_status(batch_id)
        elif self.provider_name == "anthropic":
            return self._get_anthropic_status(batch_id)
        else:
            raise ValueError(
                f"Unsupported provider for batch status: {self.provider_name}"
            )

    def retrieve_results(self, batch_id: str) -> List[BatchResult]:
        """Retrieve and parse batch results from the provider"""
        if self.provider_name == "openai":
            results_content = self._retrieve_openai_results(batch_id)
        elif self.provider_name == "anthropic":
            results_content = self._retrieve_anthropic_results(batch_id)
        else:
            raise ValueError(
                f"Unsupported provider for result retrieval: {self.provider_name}"
            )

        return self.parse_results(results_content)

    def list_batches(self, limit: int = 10) -> List[BatchJobInfo]:
        """List batch jobs for the current provider

        Args:
            limit: Maximum number of batch jobs to return

        Returns:
            List of BatchJobInfo objects with normalized batch information
        """
        if self.provider_name == "openai":
            return self._list_openai_batches(limit)
        elif self.provider_name == "anthropic":
            return self._list_anthropic_batches(limit)
        else:
            raise ValueError(
                f"Unsupported provider for listing batches: {self.provider_name}"
            )

    def get_results(
        self, batch_id: str, file_path: Optional[str] = None
    ) -> List[BatchResult]:
        """Get batch results, optionally saving raw results to a file

        Args:
            batch_id: The batch job ID
            file_path: Optional file path to save raw results. If provided,
                      raw results will be saved to this file. If not provided,
                      results are only kept in memory.

        Returns:
            List of BatchResult objects (BatchSuccess[T] or BatchError)
        """
        # Retrieve results directly to memory
        results_content = self.retrieve_results(batch_id)

        # If file path is provided, save raw results to file
        if file_path is not None:
            # Get the raw content again for saving
            if self.provider_name == "openai":
                self._download_openai_results(batch_id, file_path)
            elif self.provider_name == "anthropic":
                self._download_anthropic_results(batch_id, file_path)
            else:
                raise ValueError(
                    f"Unsupported provider for result download: {self.provider_name}"
                )

        return results_content

    def cancel_batch(self, batch_id: str) -> Dict[str, Any]:
        """Cancel a batch job

        Args:
            batch_id: The batch job ID to cancel

        Returns:
            Dict containing the cancelled batch information
        """
        if self.provider_name == "openai":
            return self._cancel_openai_batch(batch_id)
        elif self.provider_name == "anthropic":
            return self._cancel_anthropic_batch(batch_id)
        else:
            raise ValueError(
                f"Unsupported provider for batch cancellation: {self.provider_name}"
            )

    def delete_batch(self, batch_id: str) -> Dict[str, Any]:
        """Delete a batch job (only available for completed batches)

        Args:
            batch_id: The batch job ID to delete

        Returns:
            Dict containing the deletion confirmation
        """
        if self.provider_name == "openai":
            return self._delete_openai_batch(batch_id)
        elif self.provider_name == "anthropic":
            return self._delete_anthropic_batch(batch_id)
        else:
            raise ValueError(
                f"Unsupported provider for batch deletion: {self.provider_name}"
            )

    def _submit_openai_batch(
        self, file_path: str, metadata: Optional[Dict[str, Any]] = None, **kwargs
    ) -> str:
        """Submit OpenAI batch job"""
        try:
            from openai import OpenAI

            client = OpenAI()

            if metadata is None:
                metadata = {"description": "Instructor batch job"}

            with open(file_path, "rb") as f:
                batch_file = client.files.create(file=f, purpose="batch")

            batch_job = client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/chat/completions",
                completion_window=kwargs.get("completion_window", "24h"),
                metadata=metadata,
            )
            return batch_job.id
        except Exception as e:
            raise Exception(f"Failed to submit OpenAI batch: {e}")

    def _submit_anthropic_batch(
        self, file_path: str, metadata: Optional[Dict[str, Any]] = None, **kwargs
    ) -> str:
        """Submit Anthropic batch job"""
        _ = kwargs  # Unused but accepted for API consistency
        try:
            import anthropic

            client = anthropic.Anthropic()

            # Note: Anthropic doesn't support metadata in batch creation
            # but we accept it for API consistency
            if metadata:
                print(
                    f"Note: Anthropic batches don't support metadata. Ignoring: {metadata}"
                )

            # TODO: Remove beta fallback when stable API is available
            try:
                batches_client = client.messages.batches
            except AttributeError:
                batches_client = client.beta.messages.batches

            with open(file_path) as f:
                requests = [json.loads(line) for line in f if line.strip()]

            batch = batches_client.create(requests=requests)
            return batch.id
        except Exception as e:
            raise Exception(f"Failed to submit Anthropic batch: {e}")

    def _get_openai_status(self, batch_id: str) -> Dict[str, Any]:
        """Get OpenAI batch status"""
        try:
            from openai import OpenAI

            client = OpenAI()
            batch = client.batches.retrieve(batch_id)
            return {
                "id": batch.id,
                "status": batch.status,
                "created_at": batch.created_at,
                "request_counts": {
                    "total": getattr(batch.request_counts, "total", 0),
                    "completed": getattr(batch.request_counts, "completed", 0),
                    "failed": getattr(batch.request_counts, "failed", 0),
                },
            }
        except Exception as e:
            raise Exception(f"Failed to get OpenAI batch status: {e}")

    def _get_anthropic_status(self, batch_id: str) -> Dict[str, Any]:
        """Get Anthropic batch status"""
        try:
            import anthropic

            client = anthropic.Anthropic()

            # TODO: Remove beta fallback when stable API is available
            try:
                batches_client = client.messages.batches
            except AttributeError:
                batches_client = client.beta.messages.batches

            batch = batches_client.retrieve(batch_id)
            return {
                "id": batch.id,
                "status": batch.processing_status,
                "created_at": batch.created_at,
                "request_counts": getattr(batch, "request_counts", {}),
            }
        except Exception as e:
            raise Exception(f"Failed to get Anthropic batch status: {e}")

    def _retrieve_openai_results(self, batch_id: str) -> str:
        """Retrieve OpenAI batch results"""
        try:
            from openai import OpenAI

            client = OpenAI()
            batch = client.batches.retrieve(batch_id)

            if batch.status != "completed":
                raise Exception(f"Batch not completed, status: {batch.status}")

            if not batch.output_file_id:
                raise Exception("No output file available")

            file_response = client.files.content(batch.output_file_id)
            return file_response.text
        except Exception as e:
            raise Exception(f"Failed to retrieve OpenAI results: {e}")

    def _retrieve_anthropic_results(self, batch_id: str) -> str:
        """Retrieve Anthropic batch results"""
        try:
            import anthropic

            client = anthropic.Anthropic()

            # TODO: Remove beta fallback when stable API is available
            try:
                batches_client = client.messages.batches
            except AttributeError:
                batches_client = client.beta.messages.batches

            batch = batches_client.retrieve(batch_id)

            # Check for various terminal states
            if batch.processing_status in ["failed", "cancelled", "expired"]:
                raise Exception(
                    f"Batch job failed with status: {batch.processing_status}"
                )

            if batch.processing_status != "ended":
                raise Exception(
                    f"Batch not completed, status: {batch.processing_status}"
                )

            results = batches_client.results(batch_id)
            results_lines = []
            for result in results:
                results_lines.append(result.model_dump_json())

            return "\n".join(results_lines)
        except Exception as e:
            raise Exception(f"Failed to retrieve Anthropic results: {e}")

    def parse_results(self, results_content: str) -> List[BatchResult]:
        """Parse batch results from content string into Maybe-like results with custom_id tracking"""
        results: List[BatchResult] = []

        lines = results_content.strip().split("\n")
        for line in lines:
            if not line.strip():
                continue

            try:
                data = json.loads(line)
                custom_id = data.get("custom_id", "unknown")
                extracted_data = self._extract_from_response(data)

                if extracted_data:
                    try:
                        # Parse into response model
                        result = self.response_model(**extracted_data)
                        batch_result = BatchSuccess[T](
                            custom_id=custom_id, result=result
                        )
                        results.append(batch_result)
                    except Exception as e:
                        error_result = BatchError(
                            custom_id=custom_id,
                            error_type="parsing_error",
                            error_message=f"Failed to parse into {self.response_model.__name__}: {e}",
                            raw_data=extracted_data,
                        )
                        results.append(error_result)
                else:
                    # Check if this is a provider error response
                    error_message = "Unknown error"
                    error_type = "extraction_error"

                    if self.provider_name == "anthropic" and "result" in data:
                        result = data["result"]
                        if result.get("type") == "error":
                            error_info = result.get("error", {})
                            if isinstance(error_info, dict) and "error" in error_info:
                                error_details = error_info["error"]
                                error_message = error_details.get(
                                    "message", "Unknown Anthropic error"
                                )
                                error_type = error_details.get(
                                    "type", "anthropic_error"
                                )
                            else:
                                error_message = str(error_info)
                                error_type = "anthropic_error"

                    error_result = BatchError(
                        custom_id=custom_id,
                        error_type=error_type,
                        error_message=error_message,
                        raw_data=data,
                    )
                    results.append(error_result)

            except Exception as e:
                error_result = BatchError(
                    custom_id="unknown",
                    error_type="json_parse_error",
                    error_message=f"Failed to parse JSON: {e}",
                    raw_data={"raw_line": line},
                )
                results.append(error_result)

        return results

    def _extract_from_response(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract structured data from provider-specific response format"""
        try:
            if self.provider_name == "openai":
                # OpenAI JSON schema response
                content = data["response"]["body"]["choices"][0]["message"]["content"]
                return json.loads(content)

            elif self.provider_name == "anthropic":
                # Anthropic batch response format
                if "result" not in data:
                    return None

                result = data["result"]

                # Check if result is an error
                if result.get("type") == "error":
                    # Return None to indicate error, let caller handle
                    return None

                # Handle successful message result
                if result.get("type") == "succeeded" and "message" in result:
                    content = result["message"]["content"]
                    if isinstance(content, list) and len(content) > 0:
                        # Try tool_use first
                        for item in content:
                            if item.get("type") == "tool_use":
                                return item.get("input", {})

                        # Fallback to text content and parse JSON
                        for item in content:
                            if item.get("type") == "text":
                                text = item.get("text", "")
                                try:
                                    return json.loads(text)
                                except json.JSONDecodeError:
                                    continue

                return None

        except Exception:
            return None

        return None

    def _download_openai_results(self, batch_id: str, file_path: str) -> None:
        """Download OpenAI batch results to a file"""
        try:
            from openai import OpenAI

            client = OpenAI()
            batch = client.batches.retrieve(batch_id)

            if batch.status != "completed":
                raise Exception(f"Batch not completed, status: {batch.status}")

            if not batch.output_file_id:
                raise Exception("No output file available")

            file_response = client.files.content(batch.output_file_id)
            with open(file_path, "w") as f:
                f.write(file_response.text)
        except Exception as e:
            raise Exception(f"Failed to download OpenAI results: {e}")

    def _download_anthropic_results(self, batch_id: str, file_path: str) -> None:
        """Download Anthropic batch results to a file"""
        try:
            import anthropic

            client = anthropic.Anthropic()

            # TODO: Remove beta fallback when stable API is available
            try:
                batches_client = client.messages.batches
            except AttributeError:
                batches_client = client.beta.messages.batches

            batch = batches_client.retrieve(batch_id)

            # Check for various terminal states
            if batch.processing_status in ["failed", "cancelled", "expired"]:
                raise Exception(
                    f"Batch job failed with status: {batch.processing_status}"
                )

            if batch.processing_status != "ended":
                raise Exception(
                    f"Batch not completed, status: {batch.processing_status}"
                )

            results = batches_client.results(batch_id)
            with open(file_path, "w") as f:
                for result in results:
                    f.write(result.model_dump_json() + "\n")
        except Exception as e:
            raise Exception(f"Failed to download Anthropic results: {e}")

    def _list_openai_batches(self, limit: int) -> List[BatchJobInfo]:
        """List OpenAI batch jobs"""
        try:
            from openai import OpenAI

            client = OpenAI()
            batches = client.batches.list(limit=limit)

            batch_infos = []
            for batch in batches.data:
                batch_data = (
                    batch.model_dump()
                    if hasattr(batch, "model_dump")
                    else batch.__dict__
                )
                batch_info = BatchJobInfo.from_openai(batch_data)
                batch_infos.append(batch_info)

            return batch_infos
        except Exception as e:
            raise Exception(f"Failed to list OpenAI batches: {e}")

    def _list_anthropic_batches(self, limit: int) -> List[BatchJobInfo]:
        """List Anthropic batch jobs"""
        try:
            import anthropic

            client = anthropic.Anthropic()

            # TODO: Remove beta fallback when stable API is available
            try:
                batches_client = client.messages.batches
            except AttributeError:
                batches_client = client.beta.messages.batches

            response = batches_client.list(limit=limit)

            batch_infos = []
            for batch in response.data:
                batch_data = (
                    batch.model_dump()
                    if hasattr(batch, "model_dump")
                    else batch.__dict__
                )
                batch_info = BatchJobInfo.from_anthropic(batch_data)
                batch_infos.append(batch_info)

            return batch_infos
        except Exception as e:
            raise Exception(f"Failed to list Anthropic batches: {e}")

    def _cancel_openai_batch(self, batch_id: str) -> Dict[str, Any]:
        """Cancel OpenAI batch job"""
        try:
            from openai import OpenAI

            client = OpenAI()
            batch = client.batches.cancel(batch_id)
            return (
                batch.model_dump() if hasattr(batch, "model_dump") else batch.__dict__
            )
        except Exception as e:
            raise Exception(f"Failed to cancel OpenAI batch: {e}")

    def _cancel_anthropic_batch(self, batch_id: str) -> Dict[str, Any]:
        """Cancel Anthropic batch job"""
        try:
            import anthropic

            client = anthropic.Anthropic()

            # TODO: Remove beta fallback when stable API is available
            try:
                batches_client = client.messages.batches
            except AttributeError:
                batches_client = client.beta.messages.batches

            batch = batches_client.cancel(batch_id)
            return (
                batch.model_dump() if hasattr(batch, "model_dump") else batch.__dict__
            )
        except Exception as e:
            raise Exception(f"Failed to cancel Anthropic batch: {e}")

    def _delete_openai_batch(self, batch_id: str) -> Dict[str, Any]:
        """Delete OpenAI batch job"""
        # Note: OpenAI doesn't have a delete batch API endpoint
        # Batches are automatically deleted after a certain period
        _ = batch_id  # Unused but required for interface consistency
        raise NotImplementedError("OpenAI does not support batch deletion via API")

    def _delete_anthropic_batch(self, batch_id: str) -> Dict[str, Any]:
        """Delete Anthropic batch job"""
        try:
            import anthropic

            client = anthropic.Anthropic()

            # TODO: Remove beta fallback when stable API is available
            try:
                batches_client = client.messages.batches
            except AttributeError:
                batches_client = client.beta.messages.batches

            response = batches_client.delete(batch_id)
            return (
                response.model_dump()
                if hasattr(response, "model_dump")
                else response.__dict__
            )
        except Exception as e:
            raise Exception(f"Failed to delete Anthropic batch: {e}")


class BatchJob:
    """Legacy BatchJob class for backward compatibility"""

    @classmethod
    def parse_from_file(
        cls, file_path: str, response_model: type[T]
    ) -> tuple[list[T], list[dict[Any, Any]]]:
        with open(file_path) as file:
            content = file.read()
        return cls.parse_from_string(content, response_model)

    @classmethod
    def parse_from_string(
        cls, content: str, response_model: type[T]
    ) -> tuple[list[T], list[dict[Any, Any]]]:
        """Enhanced parser that works with all providers using JSON schema"""
        res: list[T] = []
        error_objs: list[dict[Any, Any]] = []

        lines = content.strip().split("\n")
        for line in lines:
            if not line.strip():
                continue

            try:
                data = json.loads(line)
                extracted_data = cls._extract_structured_data(data)

                if extracted_data:
                    try:
                        result = response_model(**extracted_data)
                        res.append(result)
                    except Exception:
                        error_objs.append(data)
                else:
                    error_objs.append(data)

            except Exception:
                error_objs.append({"error": "Failed to parse JSON", "raw_line": line})

        return res, error_objs

    @classmethod
    def _extract_structured_data(cls, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract structured data from various provider response formats"""
        try:
            # Try OpenAI JSON schema format first
            if "response" in data and "body" in data["response"]:
                choices = data["response"]["body"].get("choices", [])
                if choices:
                    message = choices[0].get("message", {})

                    # JSON schema response
                    if "content" in message:
                        content = message["content"]
                        if isinstance(content, str):
                            return json.loads(content)

                    # Tool calls (legacy)
                    if "tool_calls" in message:
                        tool_call = message["tool_calls"][0]
                        return json.loads(tool_call["function"]["arguments"])

            # Try Anthropic format
            if "result" in data and "message" in data["result"]:
                content = data["result"]["message"]["content"]
                if isinstance(content, list) and len(content) > 0:
                    # Tool use response
                    for item in content:
                        if item.get("type") == "tool_use":
                            return item.get("input", {})
                    # Text response with JSON
                    for item in content:
                        if item.get("type") == "text":
                            text = item.get("text", "")
                            return json.loads(text)

        except Exception:
            pass

        return None

    @classmethod
    def create_from_messages(
        cls,
        messages_batch: Union[
            list[list[dict[str, Any]]], Iterable[list[dict[str, Any]]]
        ],
        model: str,
        response_model: type[BaseModel],
        file_path: str,
        max_tokens: Optional[int] = 1000,
        temperature: Optional[float] = 1.0,
    ):
        """Create batch file from messages using provider detection"""
        # Detect provider from model name
        use_anthropic = "claude" in model.lower()

        # Use the new BatchProcessor for unified handling
        if use_anthropic:
            full_model = f"anthropic/{model}"
        else:
            full_model = f"openai/{model}"

        try:
            processor = BatchProcessor(full_model, response_model)
            processor.create_batch_from_messages(
                list(messages_batch), file_path, max_tokens, temperature
            )
        except Exception:
            # Fallback to legacy implementation for backward compatibility
            cls._create_legacy_format(
                messages_batch,
                model,
                response_model,
                file_path,
                max_tokens,
                temperature,
            )

    @classmethod
    def _create_legacy_format(
        cls,
        messages_batch: Union[
            list[list[dict[str, Any]]], Iterable[list[dict[str, Any]]]
        ],
        model: str,
        response_model: type[BaseModel],
        file_path: str,
        max_tokens: Optional[int] = 1000,
        temperature: Optional[float] = 1.0,
    ):
        """Legacy implementation for backward compatibility"""
        use_anthropic = "claude" in model.lower()

        if use_anthropic:
            _, kwargs = handle_response_model(
                response_model=response_model, mode=instructor.Mode.ANTHROPIC_JSON
            )
            with open(file_path, "w") as file:
                for messages in messages_batch:
                    # Format specifically for Anthropic batch API
                    request = {
                        "custom_id": str(uuid.uuid4()),
                        "params": {
                            "model": model,
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                            "messages": messages,
                            **kwargs,
                        },
                    }
                    file.write(json.dumps(request) + "\n")
        else:
            # Existing OpenAI format
            _, kwargs = handle_response_model(
                response_model=response_model, mode=instructor.Mode.TOOLS
            )
            with open(file_path, "w") as file:
                for messages in messages_batch:
                    batch_model = BatchModel(
                        custom_id=str(uuid.uuid4()),
                        body=RequestBody(
                            model=model,
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            **kwargs,
                        ),
                        method="POST",
                        url="/v1/chat/completions",
                    )
                    file.write(batch_model.model_dump_json() + "\n")


# Keep legacy models for backward compatibility
class Function(BaseModel):
    name: str
    description: str
    parameters: Any


class Tool(BaseModel):
    type: str
    function: Function


class RequestBody(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    max_tokens: Optional[int] = Field(default=1000)
    temperature: Optional[float] = Field(default=1.0)
    tools: Optional[list[Tool]]
    tool_choice: Optional[dict[str, Any]]


class BatchModel(BaseModel):
    custom_id: str
    body: RequestBody
    url: str
    method: str
