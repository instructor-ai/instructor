"""
Mistral-specific batch processing implementation.

This module contains the Mistral batch processing provider class.
"""

import io
import json
import logging
import os
from typing import Any, Optional, Union

from ..models import BatchJobInfo
from .base import BatchProvider

logger = logging.getLogger(__name__)


class MistralProvider(BatchProvider):
    """Mistral batch processing provider"""

    def submit_batch(
        self,
        file_path_or_buffer: Union[str, io.BytesIO],
        metadata: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """Submit Mistral batch job"""
        try:
            from mistralai import Mistral

            client = Mistral(api_key=os.getenv("MISTRAL_API_KEY", ""))

            logger.debug(f"Submitting batch job with metadata: {metadata}")

            # Parse the JSONL file to get requests
            if isinstance(file_path_or_buffer, str):
                logger.debug(f"Reading batch file from path: {file_path_or_buffer}")
                with open(file_path_or_buffer) as f:
                    requests = [json.loads(line) for line in f if line.strip()]
            elif isinstance(file_path_or_buffer, io.BytesIO):
                logger.debug("Reading batch file from BytesIO buffer")
                file_path_or_buffer.seek(0)
                content = file_path_or_buffer.read().decode("utf-8")
                requests = [
                    json.loads(line) for line in content.split("\n") if line.strip()
                ]
            else:
                raise ValueError(
                    f"Unsupported file_path_or_buffer type: {type(file_path_or_buffer)}"
                )

            # Check if we should use inline batching or file batching
            # Mistral supports inline batching for < 10k requests
            if len(requests) < 10000:
                logger.debug(f"Using inline batching for {len(requests)} requests")
                # Use inline batching
                batch_job = client.batch.jobs.create(
                    requests=requests,
                    model=kwargs.get("model", "mistral-small-latest"),
                    endpoint=kwargs.get("endpoint", "/v1/chat/completions"),
                    metadata=metadata or {},
                )
            else:
                logger.debug(f"Using file batching for {len(requests)} requests")
                # Upload file first for larger batches
                if isinstance(file_path_or_buffer, str):
                    with open(file_path_or_buffer, "rb") as f:
                        batch_file = client.files.upload(
                            file={"file_name": "batch.jsonl", "content": f},
                            purpose="batch",
                        )
                else:
                    file_path_or_buffer.seek(0)
                    batch_file = client.files.upload(
                        file={
                            "file_name": "batch.jsonl",
                            "content": file_path_or_buffer,
                        },
                        purpose="batch",
                    )

                # Create batch job with file reference
                batch_job = client.batch.jobs.create(
                    input_files=[batch_file.id],
                    model=kwargs.get("model", "mistral-small-latest"),
                    endpoint=kwargs.get("endpoint", "/v1/chat/completions"),
                    metadata=metadata or {},
                )

            logger.info(f"Successfully submitted batch job: {batch_job.id}")
            return batch_job.id
        except (ValueError, TypeError) as e:
            # Re-raise validation errors as-is
            logger.error(f"Validation error in Mistral batch submission: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to submit Mistral batch: {e}")
            raise RuntimeError(f"Failed to submit Mistral batch: {e}") from e

    def get_status(self, batch_id: str) -> dict[str, Any]:
        """Get Mistral batch status"""
        try:
            from mistralai import Mistral

            client = Mistral(api_key=os.getenv("MISTRAL_API_KEY", ""))
            batch = client.batch.jobs.get(job_id=batch_id)

            # Extract request counts from the batch object
            request_counts = {}
            if hasattr(batch, "total_requests"):
                request_counts["total"] = batch.total_requests
            if hasattr(batch, "succeeded_requests"):
                request_counts["succeeded"] = batch.succeeded_requests
            if hasattr(batch, "failed_requests"):
                request_counts["failed"] = batch.failed_requests

            return {
                "id": batch.id,
                "status": batch.status,
                "created_at": batch.created_at,
                "request_counts": request_counts,
            }
        except Exception as e:
            raise Exception(f"Failed to get Mistral batch status: {e}") from e

    def retrieve_results(self, batch_id: str) -> str:
        """Retrieve Mistral batch results"""
        try:
            from mistralai import Mistral

            client = Mistral(api_key=os.getenv("MISTRAL_API_KEY", ""))
            batch = client.batch.jobs.get(job_id=batch_id)

            # Check batch status
            if batch.status in ["FAILED", "TIMEOUT_EXCEEDED", "CANCELLED"]:
                raise Exception(f"Batch job failed with status: {batch.status}")

            if batch.status != "SUCCESS":
                raise Exception(f"Batch not completed, status: {batch.status}")

            # Check if all requests failed
            if hasattr(batch, "succeeded_requests") and hasattr(
                batch, "failed_requests"
            ):
                succeeded = batch.succeeded_requests
                failed = batch.failed_requests
                total = getattr(batch, "total_requests", succeeded + failed)

                if failed > 0 and succeeded == 0:
                    raise RuntimeError(
                        f"All {total} batch requests failed. No results will be available."
                    )

            if not batch.output_file:
                raise RuntimeError("Batch has no output file ID available")

            # Download the results file
            file_response = client.files.download(file_id=batch.output_file)
            return file_response.read().decode("utf-8")
        except Exception as e:
            raise Exception(f"Failed to retrieve Mistral results: {e}") from e

    def download_results(self, batch_id: str, file_path: str) -> None:
        """Download Mistral batch results to a file"""
        try:
            from mistralai import Mistral

            client = Mistral(api_key=os.getenv("MISTRAL_API_KEY", ""))
            batch = client.batch.jobs.get(job_id=batch_id)

            # Check batch status
            if batch.status in ["FAILED", "TIMEOUT_EXCEEDED", "CANCELLED"]:
                raise Exception(f"Batch job failed with status: {batch.status}")

            if batch.status != "SUCCESS":
                raise Exception(f"Batch not completed, status: {batch.status}")

            # Check if all requests failed
            if hasattr(batch, "succeeded_requests") and hasattr(
                batch, "failed_requests"
            ):
                succeeded = batch.succeeded_requests
                failed = batch.failed_requests
                total = getattr(batch, "total_requests", succeeded + failed)

                if failed > 0 and succeeded == 0:
                    raise RuntimeError(
                        f"All {total} batch requests failed. No results will be available."
                    )

            if not batch.output_file:
                raise RuntimeError("Batch has no output file ID available")

            # Download the results file
            file_response = client.files.download(file_id=batch.output_file)
            with open(file_path, "wb") as f:
                f.write(file_response.read())
        except Exception as e:
            raise Exception(f"Failed to download Mistral results: {e}") from e

    def cancel_batch(self, batch_id: str) -> dict[str, Any]:
        """Cancel Mistral batch job"""
        try:
            from mistralai import Mistral

            client = Mistral(api_key=os.getenv("MISTRAL_API_KEY", ""))
            batch = client.batch.jobs.cancel(job_id=batch_id)

            # Convert batch object to dict
            if hasattr(batch, "model_dump"):
                return batch.model_dump()
            elif hasattr(batch, "dict"):
                return batch.dict()
            else:
                # Fallback: manually extract attributes
                return {
                    "id": batch.id,
                    "status": batch.status,
                    "created_at": getattr(batch, "created_at", None),
                }
        except Exception as e:
            raise Exception(f"Failed to cancel Mistral batch: {e}") from e

    def delete_batch(self, batch_id: str) -> dict[str, Any]:
        """Delete Mistral batch job"""
        try:
            from mistralai import Mistral

            client = Mistral(api_key=os.getenv("MISTRAL_API_KEY", ""))
            # Mistral doesn't have a delete endpoint, so we'll return the batch info
            batch = client.batch.jobs.get(job_id=batch_id)
            return {
                "id": batch.id,
                "status": batch.status,
                "message": "Mistral does not support batch deletion",
            }
        except Exception as e:
            raise Exception(f"Failed to delete Mistral batch: {e}") from e

    def list_batches(self, limit: int = 10) -> list[BatchJobInfo]:
        """List Mistral batch jobs"""
        try:
            from mistralai import Mistral

            client = Mistral(api_key=os.getenv("MISTRAL_API_KEY", ""))

            # Note: Mistral's list API might have different parameters
            # Adjust based on actual SDK capabilities
            batches_response = client.batch.jobs.list()

            # Get the list of batches (limit to specified number)
            batches: list[Any] = []
            if hasattr(batches_response, "data"):
                batches_data = getattr(batches_response, "data", [])
                if isinstance(batches_data, list):
                    batches = batches_data[:limit]
            elif isinstance(batches_response, list):
                batches = batches_response[:limit]
            else:
                # Try to iterate if it's an iterable
                try:
                    batches = list(batches_response)[:limit]  # type: ignore
                except TypeError:
                    batches = [batches_response][:limit]

            result: list[BatchJobInfo] = []
            for batch in batches:
                # Convert batch to dict
                batch_dict: dict[str, Any]
                if hasattr(batch, "model_dump") and callable(batch.model_dump):  # type: ignore
                    batch_dict = batch.model_dump()  # type: ignore
                elif hasattr(batch, "dict") and callable(batch.dict):  # type: ignore
                    batch_dict = batch.dict()  # type: ignore
                elif isinstance(batch, dict):
                    batch_dict = batch
                else:
                    # Skip items that can't be converted
                    continue

                result.append(BatchJobInfo.from_mistral(batch_dict))

            return result
        except Exception as e:
            raise Exception(f"Failed to list Mistral batches: {e}") from e
