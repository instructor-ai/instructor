"""Provider system for instructor."""

from .base import BaseProvider
from .registry import ProviderRegistry

__all__ = ["BaseProvider", "ProviderRegistry"]
