"""Instructor v2 - Hierarchical mode registry system.

v2 introduces a hierarchical (Provider, ModeType) design instead of flat mode enums:

Before (v1):
    from instructor import Mode
    mode = Mode.ANTHROPIC_TOOLS  # Flat enum, 42 total

After (v2):
    from instructor.v2 import Provider, ModeType
    mode = (Provider.ANTHROPIC, ModeType.TOOLS)  # Composable, 21 enums define all combinations

Benefits:
- Composable: 15 providers × 6 mode types = any combination
- Queryable: Filter by provider OR mode type
- Extensible: Add provider = support all mode types automatically
- Provider-agnostic: Write code using ModeType.TOOLS across any provider
"""

from instructor.v2.core.handler import ModeHandler
from instructor.v2.core.mode_types import Mode, ModeType, Provider
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
    "ModeType",
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
