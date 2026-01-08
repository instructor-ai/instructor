# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

<!-- Add upcoming changes here -->

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
