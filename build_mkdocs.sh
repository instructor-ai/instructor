# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Initialize virtual environment and install dependencies
uv sync --group docs
uv run mkdocs build
