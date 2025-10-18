"""
OpenAI-specific batch processing implementation.

This module contains the OpenAI batch processing provider class.
"""

from typing import Any, Optional, Union
import io
import logging
from .base import BatchProvider
from ..models import BatchJobInfo

logger = logging.getLogger(__name__)


class OpenAIProvider(BatchProvider):
    """OpenAI batch processing provider"""

    def submit_batch(
        self,
        file_path_or_buffer: Union[str, io.BytesIO],
        metadata: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """Submit OpenAI batch job"""
        try:
            from openai import OpenAI

            client = OpenAI()

            if metadata is None:
                metadata = {"description": "Instructor batch job"}

            logger.debug(f"Submitting batch job with metadata: {metadata}")

            if isinstance(file_path_or_buffer, str):
                logger.debug(f"Creating batch file from path: {file_path_or_buffer}")
                with open(file_path_or_buffer, "rb") as f:
                    batch_file = client.files.create(file=f, purpose="batch")
            elif isinstance(file_path_or_buffer, io.BytesIO):
                logger.debug("Creating batch file from BytesIO buffer")
                file_path_or_buffer.seek(0)
                batch_file = client.files.create(
                    file=file_path_or_buffer, purpose="batch"
                )
            else:
                raise ValueError(
                    f"Unsupported file_path_or_buffer type: {type(file_path_or_buffer)}"
                )

            batch_job = client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/chat/completions",
                completion_window=kwargs.get("completion_window", "24h"),
                metadata=metadata,
            )
            logger.info(f"Successfully submitted batch job: {batch_job.id}")
            return batch_job.id
        except (ValueError, TypeError) as e:
            # Re-raise validation errors as-is
            logger.error(f"Validation error in OpenAI batch submission: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to submit OpenAI batch: {e}")
            raise RuntimeError(f"Failed to submit OpenAI batch: {e}") from e

    def get_status(self, batch_id: str) -> dict[str, Any]:
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
            raise Exception(f"Failed to get OpenAI batch status: {e}") from e

    def retrieve_results(self, batch_id: str) -> str:
        """Retrieve OpenAI batch results"""
        try:
            from openai import OpenAI
            import time

            client = OpenAI()
            batch = client.batches.retrieve(batch_id)

            if batch.status != "completed":
                raise Exception(f"Batch not completed, status: {batch.status}")

            # Check if all requests failed
            request_counts = getattr(batch, "request_counts", None)
            if request_counts:
                completed = getattr(request_counts, "completed", 0)
                failed = getattr(request_counts, "failed", 0)
                total = getattr(request_counts, "total", 0)

                if failed > 0 and completed == 0:
                    raise RuntimeError(
                        f"All {total} batch requests failed. No output file will be available. "
                    )

            if not batch.output_file_id:
                # Sometimes output file isn't immediately available, wait longer and retry more
                max_retries = 10
                for attempt in range(max_retries):
                    wait_time = min(
                        5 + attempt, 15
                    )  # Progressive backoff: 5s, 6s, 7s... up to 15s
                    print(
                        f"Output file not ready, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(wait_time)
                    batch = client.batches.retrieve(batch_id)
                    if batch.output_file_id:
                        print(f"Output file now available: {batch.output_file_id}")
                        break
                    # Check if batch failed during our wait
                    if batch.status != "completed":
                        raise Exception(
                            f"Batch status changed to {batch.status} while waiting for output file"
                        )
                    if attempt == max_retries - 1:
                        # Final attempt - provide detailed error info
                        raise RuntimeError(
                            f"No output file available after {max_retries} retries over {sum(range(5, 5 + max_retries))} seconds. "
                            f"Batch status: {batch.status}, Request counts: {getattr(batch, 'request_counts', 'unknown')}. "
                        )

            if batch.output_file_id is None:
                raise RuntimeError("Batch has no output file ID available")
            file_response = client.files.content(batch.output_file_id)
            return file_response.text
        except Exception as e:
            raise Exception(f"Failed to retrieve OpenAI results: {e}") from e

    def download_results(self, batch_id: str, file_path: str) -> None:
        """Download OpenAI batch results to a file"""
        try:
            from openai import OpenAI
            import time

            client = OpenAI()
            batch = client.batches.retrieve(batch_id)

            if batch.status != "completed":
                raise Exception(f"Batch not completed, status: {batch.status}")

            # Check if all requests failed
            request_counts = getattr(batch, "request_counts", None)
            if request_counts:
                completed = getattr(request_counts, "completed", 0)
                failed = getattr(request_counts, "failed", 0)
                total = getattr(request_counts, "total", 0)

                if failed > 0 and completed == 0:
                    raise RuntimeError(
                        f"All {total} batch requests failed. No output file will be available."
                    )

            if not batch.output_file_id:
                # Sometimes output file isn't immediately available, wait longer and retry more
                max_retries = 10
                for attempt in range(max_retries):
                    wait_time = min(
                        5 + attempt, 15
                    )  # Progressive backoff: 5s, 6s, 7s... up to 15s
                    print(
                        f"Output file not ready, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(wait_time)
                    batch = client.batches.retrieve(batch_id)
                    if batch.output_file_id:
                        print(f"Output file now available: {batch.output_file_id}")
                        break
                    # Check if batch failed during our wait
                    if batch.status != "completed":
                        raise Exception(
                            f"Batch status changed to {batch.status} while waiting for output file"
                        )
                    if attempt == max_retries - 1:
                        # Final attempt - provide detailed error info
                        raise Exception(
                            f"No output file available after {max_retries} retries over {sum(range(5, 5 + max_retries))} seconds. "
                            f"Batch status: {batch.status}, Request counts: {getattr(batch, 'request_counts', 'unknown')}."
                        )

            if batch.output_file_id is None:
                raise RuntimeError("Batch has no output file ID available")
            file_response = client.files.content(batch.output_file_id)
            with open(file_path, "w") as f:
                f.write(file_response.text)
        except Exception as e:
            raise Exception(f"Failed to download OpenAI results: {e}") from e

    def cancel_batch(self, batch_id: str) -> dict[str, Any]:
        """Cancel OpenAI batch job"""
        try:
            from openai import OpenAI

            client = OpenAI()
            batch = client.batches.cancel(batch_id)
            return batch.model_dump()
        except Exception as e:
            raise Exception(f"Failed to cancel OpenAI batch: {e}") from e

    def delete_batch(self, batch_id: str) -> dict[str, Any]:
        """Delete OpenAI batch job"""
        try:
            from openai import OpenAI

            client = OpenAI()
            # OpenAI doesn't have a delete endpoint, so we'll return the batch info
            batch = client.batches.retrieve(batch_id)
            return {
                "id": batch.id,
                "status": batch.status,
                "message": "OpenAI does not support batch deletion",
            }
        except Exception as e:
            raise Exception(f"Failed to delete OpenAI batch: {e}") from e

    def list_batches(self, limit: int = 10) -> list[BatchJobInfo]:
        """List OpenAI batch jobs"""
        try:
            from openai import OpenAI

            client = OpenAI()
            batches = client.batches.list(limit=limit)
            return [
                BatchJobInfo.from_openai(batch.model_dump()) for batch in batches.data
            ]
        except Exception as e:
            raise Exception(f"Failed to list OpenAI batches: {e}") from e
