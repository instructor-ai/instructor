"""
Anthropic-specific batch processing implementation.

This module contains the Anthropic batch processing provider class.
"""

import json
from typing import Any, Optional, Union
import io
import logging
from .base import BatchProvider
from ..models import BatchJobInfo

logger = logging.getLogger(__name__)


def _get_anthropic_batches_client(client):
    """
    Get the Anthropic batches client, preferring stable API over beta.

    The stable API (client.messages.batches) was introduced in a later version
    of the anthropic SDK. This function provides a fallback to the beta API
    (client.beta.messages.batches) for backwards compatibility.

    Args:
        client: An anthropic.Anthropic client instance

    Returns:
        The batches client (either stable or beta)

    Note:
        Once the stable API is widely available, this fallback can be removed.
        Check Anthropic SDK documentation for when the stable API became available.
    """
    try:
        # Try stable API first
        return client.messages.batches
    except AttributeError:
        # Fallback to beta API for backwards compatibility
        logger.debug(
            "Using Anthropic beta.batches API. "
            "Consider upgrading anthropic SDK for stable API access."
        )
        return client.beta.messages.batches


class AnthropicProvider(BatchProvider):
    """Anthropic batch processing provider"""

    def submit_batch(
        self,
        file_path_or_buffer: Union[str, io.BytesIO],
        metadata: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """Submit Anthropic batch job"""
        _ = kwargs  # Unused but accepted for API consistency
        try:
            import anthropic

            client = anthropic.Anthropic()

            # Note: Anthropic doesn't support metadata in batch creation
            # but we accept it for API consistency
            if metadata:
                logger.warning(
                    f"Anthropic batches don't support metadata. Ignoring: {metadata}"
                )

            batches_client = _get_anthropic_batches_client(client)

            if isinstance(file_path_or_buffer, str):
                with open(file_path_or_buffer) as f:
                    requests = [json.loads(line) for line in f if line.strip()]
            elif isinstance(file_path_or_buffer, io.BytesIO):
                file_path_or_buffer.seek(0)
                content = file_path_or_buffer.read().decode("utf-8")
                requests = [
                    json.loads(line) for line in content.split("\n") if line.strip()
                ]
            else:
                raise ValueError(
                    f"Unsupported file_path_or_buffer type: {type(file_path_or_buffer)}"
                )

            batch = batches_client.create(requests=requests)
            return batch.id
        except (ValueError, TypeError) as e:
            # Re-raise validation errors as-is
            logger.error(f"Validation error in Anthropic batch submission: {e}")
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to submit Anthropic batch: {e}") from e

    def get_status(self, batch_id: str) -> dict[str, Any]:
        """Get Anthropic batch status"""
        try:
            import anthropic

            client = anthropic.Anthropic()

            batches_client = _get_anthropic_batches_client(client)
            batch = batches_client.retrieve(batch_id)
            return {
                "id": batch.id,
                "status": batch.processing_status,
                "created_at": batch.created_at,
                "request_counts": getattr(batch, "request_counts", {}),
            }
        except Exception as e:
            raise Exception(f"Failed to get Anthropic batch status: {e}") from e

    def retrieve_results(self, batch_id: str) -> str:
        """Retrieve Anthropic batch results"""
        try:
            import anthropic

            client = anthropic.Anthropic()

            batches_client = _get_anthropic_batches_client(client)
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

            # Check if all requests failed
            request_counts = getattr(batch, "request_counts", None)
            if request_counts:
                succeeded = getattr(request_counts, "succeeded", 0)
                errored = getattr(request_counts, "errored", 0)
                total = getattr(request_counts, "total", 0)

                if errored > 0 and succeeded == 0:
                    raise RuntimeError(
                        f"All {total} batch requests failed. No results will be available."
                    )

            results = batches_client.results(batch_id)
            results_lines = []
            for result in results:
                results_lines.append(result.model_dump_json())

            return "\n".join(results_lines)
        except Exception as e:
            raise Exception(f"Failed to retrieve Anthropic results: {e}") from e

    def download_results(self, batch_id: str, file_path: str) -> None:
        """Download Anthropic batch results to a file"""
        try:
            import anthropic

            client = anthropic.Anthropic()

            batches_client = _get_anthropic_batches_client(client)
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

            # Check if all requests failed
            request_counts = getattr(batch, "request_counts", None)
            if request_counts:
                succeeded = getattr(request_counts, "succeeded", 0)
                errored = getattr(request_counts, "errored", 0)
                total = getattr(request_counts, "total", 0)

                if errored > 0 and succeeded == 0:
                    raise RuntimeError(
                        f"All {total} batch requests failed. No results will be available."
                    )

            results = batches_client.results(batch_id)
            with open(file_path, "w") as f:
                for result in results:
                    f.write(result.model_dump_json() + "\n")
        except Exception as e:
            raise Exception(f"Failed to download Anthropic results: {e}") from e

    def cancel_batch(self, batch_id: str) -> dict[str, Any]:
        """Cancel Anthropic batch job"""
        try:
            import anthropic

            client = anthropic.Anthropic()

            batches_client = _get_anthropic_batches_client(client)
            batch = batches_client.cancel(batch_id)
            return batch.model_dump()
        except Exception as e:
            raise Exception(f"Failed to cancel Anthropic batch: {e}") from e

    def delete_batch(self, batch_id: str) -> dict[str, Any]:
        """Delete Anthropic batch job"""
        try:
            import anthropic

            client = anthropic.Anthropic()

            batches_client = _get_anthropic_batches_client(client)
            batch = batches_client.retrieve(batch_id)
            return {
                "id": batch.id,
                "status": batch.processing_status,
                "message": "Anthropic does not support batch deletion",
            }
        except Exception as e:
            raise Exception(f"Failed to delete Anthropic batch: {e}") from e

    def list_batches(self, limit: int = 10) -> list[BatchJobInfo]:
        """List Anthropic batch jobs"""
        try:
            import anthropic

            client = anthropic.Anthropic()

            batches_client = _get_anthropic_batches_client(client)
            batches = batches_client.list(limit=limit)
            return [
                BatchJobInfo.from_anthropic(batch.model_dump())
                for batch in batches.data
            ]
        except Exception as e:
            raise Exception(f"Failed to list Anthropic batches: {e}") from e
