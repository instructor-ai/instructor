# AGENT.md - Documentation

## Commands
- Serve docs locally: `uv run mkdocs serve`
- Build docs: `./build_mkdocs.sh` or `uv run mkdocs build`
- Install doc deps: `uv pip install -e ".[docs]"`
- Test examples: `uv run pytest docs/ --examples`

## Structure
- **Core docs**: `concepts/`, `integrations/`, `examples/`
- **Learning path**: `getting-started.md` → `learning/` → `tutorials/`
- **API reference**: Auto-generated from docstrings via `mkdocstrings`
- **Blog**: `blog/posts/` for announcements and deep-dives
- **Templates**: `templates/` for new docs (provider, concept, cookbook)

## Writing Guidelines
- **Reading level**: Grade 10 (from .cursor/rules)
- **Code examples**: Must be runnable with complete imports
- **Progressive complexity**: Simple → advanced concepts
- **Provider docs**: Follow `templates/` patterns
- **Navigation**: Update `mkdocs.yml` for new pages

## Key Files
- `mkdocs.yml` - Site configuration and navigation
- `hooks/` - Custom processing (hide_lines.py removes `# <%hide%>` markers)
- `overrides/` - Custom theme elements
- `javascripts/` - Client-side enhancements
