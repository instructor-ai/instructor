"""
Base provider class for batch processing.

This module defines the abstract base class that all batch providers must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Union
import io
import logging
from ..models import BatchJobInfo

logger = logging.getLogger(__name__)


class BatchProvider(ABC):
    """Abstract base class for batch processing providers"""

    @abstractmethod
    def submit_batch(
        self,
        file_path_or_buffer: Union[str, io.BytesIO],
        metadata: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """Submit a batch job and return the job ID"""
        pass

    @abstractmethod
    def get_status(self, batch_id: str) -> dict[str, Any]:
        """Get the status of a batch job"""
        pass

    @abstractmethod
    def retrieve_results(self, batch_id: str) -> str:
        """Retrieve batch results as a string"""
        pass

    @abstractmethod
    def download_results(self, batch_id: str, file_path: str) -> None:
        """Download batch results to a file"""
        pass

    @abstractmethod
    def cancel_batch(self, batch_id: str) -> dict[str, Any]:
        """Cancel a batch job"""
        pass

    @abstractmethod
    def delete_batch(self, batch_id: str) -> dict[str, Any]:
        """Delete a batch job"""
        pass

    @abstractmethod
    def list_batches(self, limit: int = 10) -> list[BatchJobInfo]:
        """List batch jobs"""
        pass
