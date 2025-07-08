"""
Anthropic-specific batch processing implementation.

This module contains the Anthropic batch processing provider class.
"""

import json
from typing import Any, Optional
from .base import BatchProvider
from ..models import BatchJobInfo


class AnthropicProvider(BatchProvider):
    """Anthropic batch processing provider"""

    def submit_batch(
        self, file_path: str, metadata: Optional[dict[str, Any]] = None, **kwargs
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
            raise Exception(f"Failed to submit Anthropic batch: {e}") from e

    def get_status(self, batch_id: str) -> dict[str, Any]:
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
            raise Exception(f"Failed to get Anthropic batch status: {e}") from e

    def retrieve_results(self, batch_id: str) -> str:
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
            raise Exception(f"Failed to retrieve Anthropic results: {e}") from e

    def download_results(self, batch_id: str, file_path: str) -> None:
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
            raise Exception(f"Failed to download Anthropic results: {e}") from e

    def cancel_batch(self, batch_id: str) -> dict[str, Any]:
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
            return batch.model_dump()
        except Exception as e:
            raise Exception(f"Failed to cancel Anthropic batch: {e}") from e

    def delete_batch(self, batch_id: str) -> dict[str, Any]:
        """Delete Anthropic batch job"""
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
                "message": "Anthropic does not support batch deletion",
            }
        except Exception as e:
            raise Exception(f"Failed to delete Anthropic batch: {e}") from e

    def list_batches(self, limit: int = 10) -> list[BatchJobInfo]:
        """List Anthropic batch jobs"""
        try:
            import anthropic

            client = anthropic.Anthropic()

            # TODO: Remove beta fallback when stable API is available
            try:
                batches_client = client.messages.batches
            except AttributeError:
                batches_client = client.beta.messages.batches

            batches = batches_client.list(limit=limit)
            return [
                BatchJobInfo.from_anthropic(batch.model_dump())
                for batch in batches.data
            ]
        except Exception as e:
            raise Exception(f"Failed to list Anthropic batches: {e}") from e
