"""
OpenAI-specific batch processing implementation.

This module contains the OpenAI batch processing provider class.
"""

from typing import Any, Optional
from .base import BatchProvider
from ..models import BatchJobInfo


class OpenAIProvider(BatchProvider):
    """OpenAI batch processing provider"""

    def submit_batch(
        self, file_path: str, metadata: Optional[dict[str, Any]] = None, **kwargs
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
            raise Exception(f"Failed to submit OpenAI batch: {e}") from e

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

            client = OpenAI()
            batch = client.batches.retrieve(batch_id)

            if batch.status != "completed":
                raise Exception(f"Batch not completed, status: {batch.status}")

            if not batch.output_file_id:
                raise Exception("No output file available")

            file_response = client.files.content(batch.output_file_id)
            return file_response.text
        except Exception as e:
            raise Exception(f"Failed to retrieve OpenAI results: {e}") from e

    def download_results(self, batch_id: str, file_path: str) -> None:
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
