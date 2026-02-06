"""
Utility functions for batch processing.

This module contains helper functions for filtering, extracting, and manipulating
batch results.
"""

from .models import BatchResult, BatchSuccess, BatchError, T


def filter_successful(results: list[BatchResult]) -> list[BatchSuccess[T]]:
    """Filter to only successful results"""
    return [r for r in results if r.success]


def filter_errors(results: list[BatchResult]) -> list[BatchError]:
    """Filter to only error results"""
    return [r for r in results if not r.success]


def extract_results(results: list[BatchResult]) -> list[T]:
    """Extract just the result objects from successful results"""
    return [r.result for r in results if r.success]


def get_results_by_custom_id(results: list[BatchResult]) -> dict[str, BatchResult]:
    """Create a dictionary mapping custom_id to results"""
    return {r.custom_id: r for r in results}
