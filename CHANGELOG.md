# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

<!-- Add upcoming changes here -->

## [1.14.4] - 2026-01-16

### Changed
- Simplified `JsonCompleteness` by using `jiter` parsing and a sibling-based completeness heuristic (#2000)

### Fixed
- Fixed Responses API retries crashing on reasoning items by skipping non-tool-call items in `reask_responses_tools` (#2002)
- Fixed Google GenAI dict-style `config` handling to preserve `labels` and other settings like `cached_content` and `thinking_config` (#2005)
- Fixed Google GenAI `safety_settings` causing `400 INVALID_ARGUMENT` when requests include image content by using image-specific harm categories when needed (#2007, #1773)
- Fixed `create_with_completion()` crashing when using `list[T]` response models by preserving `_raw_response` on list outputs, and hardened optional `vertexai` imports (#2011, #1303)

## [1.14.3] - 2026-01-13

### Added
- Completeness-based validation for Partial streaming - only validates JSON structures that are structurally complete (#1999)
- New `JsonCompleteness` class in `instructor/dsl/json_tracker.py` for tracking JSON completeness during streaming (#1999)

### Fixed
- Fixed Stream objects crashing reask handlers when using streaming with `max_retries > 1` (#1992)
- Field constraints (`min_length`, `max_length`, `ge`, `le`, etc.) now work correctly during streaming (#1999)

### Deprecated
- `PartialLiteralMixin` is now deprecated - completeness-based validation handles Literal/Enum types automatically (#1999)

## [1.14.2] - 2026-01-13

### Fixed
- Fixed model validators crashing during partial streaming by skipping them until streaming completes (#1994)
- Fixed infinite recursion with self-referential models in Partial (e.g., TreeNode with children: List["TreeNode"]) (#1997)

### Added
- Added `PartialLiteralMixin` documentation for handling Literal/Enum types during streaming (#1994)
- Added final validation against original model after streaming completes to enforce required fields (#1994)
- Added tests for recursive Partial models (#1997)

## [1.14.1] - 2026-01-08

### Fixed
- Added support for cached_content in Google Gemini context caching (#1987)

## [1.14.0] - 2026-01-08

### Added
- Pre-commit hook to auto-export requirements.txt for build consistency

### Changed
- Standardized provider factory methods across codebase for improved consistency
- Standardized provider imports throughout documentation
- Audited and standardized exception handling throughout the instructor library

### Fixed
- Fixed build issues with requirements.txt regeneration from pyproject.toml
- Fixed provider functionality issue (#1914)

### Documentation
- Comprehensive documentation audit and SEO optimization improvements (#1944)
- Updated documentation for responses API mode (#1946)
- Enhanced README with PydanticAI promotion and clear feature distinctions
- Removed incorrect model reference in client.create extraction example (#1951)
- Fixed image base URLs in Jupyter notebook tutorials (#1922)

## [1.13.0] - Previous Release

For changes in earlier versions, see the [git history](https://github.com/instructor-ai/instructor/releases).
