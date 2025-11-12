"""Instructor v2 - Mode registry system.

v2 uses a registry system that maps Mode enum values directly to handlers.
This replaces the hardcoded dictionaries in v1 with a dynamic, extensible system.

Usage:
    from instructor import Mode
    from instructor.v2 import from_anthropic

    client = from_anthropic(anthropic_client, mode=Mode.ANTHROPIC_TOOLS)

Benefits:
- Dynamic registration: Modes register themselves via decorators
- Lazy loading: Handlers loaded only when used
- Extensible: Easy to add new modes without modifying core
- Type-safe: Protocols ensure handler compatibility
"""

from instructor import Mode, Provider
from instructor.v2.core.handler import ModeHandler
from instructor.v2.core.protocols import ReaskHandler, RequestHandler, ResponseParser
from instructor.v2.core.registry import ModeHandlers, ModeRegistry, mode_registry

# Import providers (will auto-register modes)
try:
    from instructor.v2.providers.anthropic import from_anthropic
except ImportError:
    from_anthropic = None  # type: ignore

__all__ = [
    # Core types
    "Provider",
    "Mode",
    # Registry
    "mode_registry",
    "ModeRegistry",
    "ModeHandlers",
    # Handler base class
    "ModeHandler",
    # Protocols
    "RequestHandler",
    "ReaskHandler",
    "ResponseParser",
    # Providers
    "from_anthropic",
]
